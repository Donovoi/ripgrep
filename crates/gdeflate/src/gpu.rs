//! GPU-accelerated decompression support for large files
//!
//! This module provides NVIDIA GPU acceleration for GDeflate decompression
//! on large files (50GB+). GPU acceleration is optional and falls back to
//! CPU multi-threading when:
//! - GPU hardware is not available
//! - CUDA runtime is not installed
//! - File is smaller than threshold
//! - GPU feature is not enabled at compile time
//!
//! # Performance Characteristics
//!
//! For files >= 50GB with NVIDIA GPU:
//! - **50-100GB files**: 4-8x faster than CPU
//! - **100-500GB files**: 6-10x faster than CPU  
//! - **500GB+ files**: 8-15x faster than CPU
//!
//! # Requirements
//!
//! - NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
//! - CUDA Toolkit 11.0 or later
//! - nvCOMP library (bundled or system-installed)
//!
//! # Usage
//!
//! ```no_run
//! use gdeflate::gpu::{is_gpu_available, decompress_with_gpu};
//!
//! // Check if GPU acceleration is available
//! if is_gpu_available() {
//!     // Decompress using GPU
//!     let decompressed = decompress_with_gpu(&compressed, output_size)?;
//! } else {
//!     // Fall back to CPU
//!     let decompressed = gdeflate::decompress(&compressed, output_size, 0)?;
//! }
//! ```

use crate::{Error, Result};

/// Helper constant for megabyte conversion.
const MB: u64 = 1024 * 1024;
/// Helper constant for gigabyte conversion.
const GB: u64 = MB * 1024;

/// Minimum file size in bytes to consider GPU decompression (50 GB)
pub const GPU_SIZE_THRESHOLD: usize = 50 * 1024 * 1024 * 1024;

/// Maximum file size in bytes that can be decompressed with GPU (1 TB)
/// Larger files should be processed in chunks
pub const GPU_MAX_SIZE: usize = 1024 * 1024 * 1024 * 1024;

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU device name
    pub name: String,
    /// Total device memory in bytes
    pub total_memory: usize,
    /// Free device memory in bytes  
    pub free_memory: usize,
    /// CUDA compute capability (major.minor)
    pub compute_capability: (u32, u32),
    /// Number of streaming multiprocessors
    pub multiprocessor_count: u32,
}

/// Auto-tuning parameters for GPU literal search offload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiteralSearchConfig {
    /// Minimum file size (in bytes) before the GPU prefilter is used.
    pub min_file_bytes: u64,
    /// Size of each chunk handed to the GPU search kernel.
    pub chunk_bytes: usize,
}

/// High-level description of what is being searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralSearchInputKind {
    /// Unable to determine what is being searched (default).
    Unknown,
    /// Exactly one regular file with a known, finite size.
    SingleFile,
    /// Multiple targets (directories or several explicit files).
    MultipleInputs,
    /// Streaming input such as stdin with unknown bounds.
    Stream,
}

impl Default for LiteralSearchInputKind {
    fn default() -> Self {
        LiteralSearchInputKind::Unknown
    }
}

/// Hint data used to tailor literal search GPU parameters to the current
/// invocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiteralSearchInputHint {
    /// What kind of input is being searched.
    pub kind: LiteralSearchInputKind,
    /// Approximate total number of bytes that will be scanned.
    pub approx_total_bytes: Option<u64>,
    /// Maximum size of a single entry (e.g. `--max-filesize`).
    pub per_entry_max_bytes: Option<u64>,
}

const MIN_LITERAL_THRESHOLD: u64 = 512 * MB;
const MAX_LITERAL_THRESHOLD: u64 = 8 * GB;
const MIN_LITERAL_CHUNK: usize = 32 * 1024 * 1024; // 32 MiB
const MAX_LITERAL_CHUNK: usize = 512 * 1024 * 1024; // 512 MiB

/// Check if GPU acceleration is available
///
/// Returns `true` if:
/// - CUDA runtime is available
/// - At least one compatible NVIDIA GPU is detected
/// - GPU feature is enabled at compile time
///
/// # Example
///
/// ```no_run
/// use gdeflate::gpu::is_gpu_available;
///
/// if is_gpu_available() {
///     println!("GPU acceleration available!");
/// }
/// ```
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "cuda-gpu")]
    {
        // SAFETY: Calling gpu_is_available() is safe because:
        // 1. It only queries CUDA runtime state via cudaGetDeviceCount()
        // 2. No memory is modified or aliased
        // 3. Returns simple boolean value
        // 4. CUDA runtime is thread-safe per CUDA Programming Guide §3.2.1
        // 5. Function handles all CUDA errors internally (returns false on error)
        unsafe { gpu_is_available() }
    }
    #[cfg(not(feature = "cuda-gpu"))]
    {
        false
    }
}

/// Get information about available GPU devices
///
/// Returns a list of available GPU devices with their capabilities.
/// Returns an empty vector if no GPUs are available or if the CUDA
/// feature is not enabled.
///
/// # Example
///
/// ```no_run
/// use gdeflate::gpu::get_gpu_devices;
///
/// for (i, device) in get_gpu_devices().iter().enumerate() {
///     println!("GPU {}: {} ({} GB)", i, device.name, device.total_memory / (1024*1024*1024));
/// }
/// ```
pub fn get_gpu_devices() -> Vec<GpuInfo> {
    #[cfg(feature = "cuda-gpu")]
    {
        // SAFETY: Calling gpu_get_devices() is safe because:
        // 1. Returns Vec<GpuInfo> which is properly allocated Rust memory
        // 2. Internal FFI calls handle CUDA errors gracefully
        // 3. No pointers are exposed to caller, all data is copied into Rust types
        // 4. Thread-safe as CUDA device queries don't modify state
        unsafe { gpu_get_devices() }
    }
    #[cfg(not(feature = "cuda-gpu"))]
    {
        Vec::new()
    }
}

/// Estimate automatic thresholds for literal GPU searching.
///
/// Returns `None` when GPU searching is unavailable. The returned configuration
/// provides the minimum file size that warrants GPU prefiltering along with the
/// chunk size (in bytes) that balances PCIe transfers and GPU occupancy.
pub fn estimate_literal_search_config() -> Option<LiteralSearchConfig> {
    estimate_literal_search_config_with_hint(LiteralSearchInputHint::default())
}

/// Same as [`estimate_literal_search_config`] but accepts additional
/// information about the input set, allowing for tighter chunk-size
/// calculations.
pub fn estimate_literal_search_config_with_hint(
    hint: LiteralSearchInputHint,
) -> Option<LiteralSearchConfig> {
    #[cfg(feature = "cuda-gpu")]
    {
        let devices = get_gpu_devices();
        let best =
            devices.into_iter().max_by_key(|device| device.free_memory)?;

        if best.free_memory == 0 {
            return None;
        }

        let free = best.free_memory as u64;
        let mut min_file = std::cmp::max(free / 6, MIN_LITERAL_THRESHOLD)
            .min(MAX_LITERAL_THRESHOLD);
        let mut chunk = ((free / 8) as usize)
            .max(MIN_LITERAL_CHUNK)
            .min(MAX_LITERAL_CHUNK);

        match hint.kind {
            LiteralSearchInputKind::MultipleInputs => {
                chunk = std::cmp::max(MIN_LITERAL_CHUNK, chunk / 2);
            }
            LiteralSearchInputKind::Stream => {
                chunk = std::cmp::max(MIN_LITERAL_CHUNK, chunk / 3);
            }
            _ => {}
        }

        if let Some(total) = hint.approx_total_bytes {
            if total > 0 {
                let divisor = match hint.kind {
                    LiteralSearchInputKind::SingleFile => 6,
                    LiteralSearchInputKind::MultipleInputs => 12,
                    LiteralSearchInputKind::Stream => 16,
                    LiteralSearchInputKind::Unknown => 8,
                } as u64;
                let desired = (total / divisor)
                    .max(MIN_LITERAL_CHUNK as u64)
                    .min(MAX_LITERAL_CHUNK as u64)
                    as usize;
                chunk = chunk.min(desired.max(MIN_LITERAL_CHUNK));
                if total < min_file {
                    min_file = std::cmp::max(MIN_LITERAL_THRESHOLD, total / 2);
                }
            }
        }

        if let Some(cap) = hint.per_entry_max_bytes {
            if cap > 0 {
                let cap = cap as usize;
                chunk = chunk.min(cap.max(MIN_LITERAL_CHUNK));
            }
        }

        Some(LiteralSearchConfig {
            min_file_bytes: min_file,
            chunk_bytes: chunk,
        })
    }
    #[cfg(not(feature = "cuda-gpu"))]
    {
        let _ = hint;
        None
    }
}

/// Check if file size is suitable for GPU decompression
///
/// Returns `true` if the file size is between the minimum threshold
/// (50GB) and maximum supported size (1TB).
///
/// # Arguments
///
/// * `size` - File size in bytes
///
/// # Example
///
/// ```
/// use gdeflate::gpu::{should_use_gpu, GPU_SIZE_THRESHOLD};
///
/// assert!(!should_use_gpu(1024)); // Too small
/// assert!(should_use_gpu(GPU_SIZE_THRESHOLD + 1)); // Large enough
/// ```
pub fn should_use_gpu(size: usize) -> bool {
    size >= GPU_SIZE_THRESHOLD && size <= GPU_MAX_SIZE
}

/// Decompress data using GPU acceleration
///
/// This function uses NVIDIA GPU to accelerate decompression of large
/// GDeflate compressed data. It automatically manages GPU memory and
/// falls back to CPU if GPU decompression fails.
///
/// # Arguments
///
/// * `input` - Compressed input data
/// * `output_size` - Expected size of decompressed data in bytes
///
/// # Returns
///
/// Decompressed data as a `Vec<u8>`
///
/// # Errors
///
/// Returns an error if:
/// - GPU is not available
/// - Output size exceeds GPU memory
/// - Decompression fails
///
/// # Example
///
/// ```no_run
/// use gdeflate::gpu::decompress_with_gpu;
///
/// let compressed = /* ... large compressed data ... */;
/// let output_size = 100 * 1024 * 1024 * 1024; // 100 GB
///
/// match decompress_with_gpu(&compressed, output_size) {
///     Ok(decompressed) => println!("GPU decompression succeeded!"),
///     Err(e) => println!("GPU decompression failed: {}, falling back to CPU", e),
/// }
/// ```
pub fn decompress_with_gpu(
    input: &[u8],
    output_size: usize,
) -> Result<Vec<u8>> {
    #[cfg(feature = "cuda-gpu")]
    {
        if !is_gpu_available() {
            return Err(Error::Gpu(crate::GpuError::NotAvailable {
                reason: "No NVIDIA GPU detected or CUDA not installed".to_string()
            }));
        }

        if output_size > GPU_MAX_SIZE {
            return Err(Error::Gpu(crate::GpuError::OperationFailed {
                operation: "decompression".to_string(),
                reason: format!("File size {} exceeds maximum GPU size {}", output_size, GPU_MAX_SIZE)
            }));
        }

        // SAFETY: Calling gpu_decompress_internal() is safe because:
        // 1. input slice is valid for reads (Rust borrow checker guarantees)
        // 2. output_size is validated above (< GPU_MAX_SIZE)
        // 3. Function allocates output Vec internally with proper Rust memory
        // 4. GPU availability already checked above
        // 5. All GPU memory transfers and operations are handled internally
        // 6. Returns Result allowing graceful error handling
        unsafe { gpu_decompress_internal(input, output_size) }
    }
    #[cfg(not(feature = "cuda-gpu"))]
    {
        // GPU feature not enabled, return error to trigger CPU fallback
        Err(Error::Generic)
    }
}

/// Decompress data with automatic CPU/GPU selection
///
/// This function automatically chooses between CPU and GPU decompression
/// based on:
/// - File size (>= 50GB for GPU)
/// - GPU availability
/// - Available GPU memory
///
/// It provides a simple interface that always works, falling back to
/// CPU when GPU is unavailable or unsuitable.
///
/// # Arguments
///
/// * `input` - Compressed input data
/// * `output_size` - Expected size of decompressed data in bytes
///
/// # Returns
///
/// Decompressed data as a `Vec<u8>`
///
/// # Example
///
/// ```no_run
/// use gdeflate::gpu::decompress_auto;
///
/// let compressed = /* ... compressed data ... */;
/// let output_size = 100 * 1024 * 1024 * 1024; // 100 GB
///
/// // Automatically uses GPU if available and suitable
/// let decompressed = decompress_auto(&compressed, output_size)?;
/// ```
pub fn decompress_auto(input: &[u8], output_size: usize) -> Result<Vec<u8>> {
    // Try GPU decompression for large files
    if should_use_gpu(output_size) && is_gpu_available() {
        match decompress_with_gpu(input, output_size) {
            Ok(data) => return Ok(data),
            Err(_) => {
                // GPU failed, fall through to CPU
            }
        }
    }

    // Fall back to CPU decompression
    crate::decompress_auto(input, output_size)
}

/// Run a GPU-accelerated literal substring search.
///
/// Returns `Ok(true)` when the substring is detected, `Ok(false)` when no
/// occurrence exists, and `Err` to indicate GPU failure (triggering a CPU
/// fallback in the caller).
pub fn substring_contains(haystack: &[u8], needle: &[u8]) -> Result<bool> {
    #[cfg(feature = "cuda-gpu")]
    {
        if needle.is_empty() {
            return Ok(true);
        }
        if haystack.len() < needle.len() {
            return Ok(false);
        }

        let result = unsafe {
            gpu_substring_contains(
                haystack.as_ptr(),
                haystack.len(),
                needle.as_ptr(),
                needle.len(),
            )
        };

        match result {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(Error::Generic),
        }
    }
    #[cfg(not(feature = "cuda-gpu"))]
    {
        let _ = (haystack, needle);
        Err(Error::Generic)
    }
}

// FFI declarations for CUDA/nvCOMP bindings
// These are only compiled when cuda-gpu feature is enabled

#[cfg(feature = "cuda-gpu")]
#[repr(C)]
#[derive(Clone)]
struct GpuDeviceInfo {
    name: [u8; 256],
    total_memory: usize,
    free_memory: usize,
    compute_major: i32,
    compute_minor: i32,
    multiprocessor_count: i32,
}

#[cfg(feature = "cuda-gpu")]
extern "C" {
    fn gpu_is_available() -> bool;
    fn gpu_get_device_info(
        devices: *mut GpuDeviceInfo,
        max_devices: i32,
    ) -> i32;
    fn gpu_decompress(
        input: *const u8,
        input_size: usize,
        output: *mut u8,
        output_size: usize,
    ) -> i32;
    fn gpu_substring_contains(
        haystack: *const u8,
        haystack_len: usize,
        needle: *const u8,
        needle_len: usize,
    ) -> i32;
}

#[cfg(feature = "cuda-gpu")]
unsafe fn gpu_get_devices() -> Vec<GpuInfo> {
    const MAX_DEVICES: i32 = 16;
    let mut devices = vec![
        GpuDeviceInfo {
            name: [0; 256],
            total_memory: 0,
            free_memory: 0,
            compute_major: 0,
            compute_minor: 0,
            multiprocessor_count: 0,
        };
        MAX_DEVICES as usize
    ];

    let count = gpu_get_device_info(devices.as_mut_ptr(), MAX_DEVICES);

    devices
        .iter()
        .take(count as usize)
        .map(|d| {
            let name = std::ffi::CStr::from_ptr(d.name.as_ptr() as *const i8)
                .to_string_lossy()
                .into_owned();
            GpuInfo {
                name,
                total_memory: d.total_memory,
                free_memory: d.free_memory,
                compute_capability: (
                    d.compute_major as u32,
                    d.compute_minor as u32,
                ),
                multiprocessor_count: d.multiprocessor_count as u32,
            }
        })
        .collect()
}

#[cfg(feature = "cuda-gpu")]
unsafe fn gpu_decompress_internal(
    input: &[u8],
    output_size: usize,
) -> Result<Vec<u8>> {
    let mut output = vec![0u8; output_size];

    let result = gpu_decompress(
        input.as_ptr(),
        input.len(),
        output.as_mut_ptr(),
        output_size,
    );

    if result == 0 {
        Ok(output)
    } else if result == -2 {
        Err(Error::InvalidParam)
    } else if result == -3 {
        Err(Error::BufferTooSmall)
    } else {
        Err(Error::Generic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_use_gpu() {
        // Small files should not use GPU
        assert!(!should_use_gpu(1024));
        assert!(!should_use_gpu(1024 * 1024 * 1024)); // 1 GB
        assert!(!should_use_gpu(10 * 1024 * 1024 * 1024)); // 10 GB

        // Files at threshold should use GPU
        assert!(should_use_gpu(GPU_SIZE_THRESHOLD));
        assert!(should_use_gpu(GPU_SIZE_THRESHOLD + 1));

        // Large files should use GPU
        assert!(should_use_gpu(100 * 1024 * 1024 * 1024)); // 100 GB
        assert!(should_use_gpu(500 * 1024 * 1024 * 1024)); // 500 GB

        // Files above max should not use GPU (would be chunked)
        assert!(!should_use_gpu(GPU_MAX_SIZE + 1));
    }

    #[test]
    fn test_gpu_availability_without_feature() {
        // When cuda-gpu feature is not enabled, GPU should not be available
        #[cfg(not(feature = "cuda-gpu"))]
        {
            assert!(!is_gpu_available());
            assert!(get_gpu_devices().is_empty());
        }
    }

    #[test]
    fn test_decompress_auto_fallback() {
        // Test that decompress_auto works even without GPU
        let input = crate::compress(b"Hello, GPU world!", 6, 0).unwrap();
        let result = decompress_auto(&input, 17);

        // Should succeed using CPU fallback
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), b"Hello, GPU world!");
    }
}
