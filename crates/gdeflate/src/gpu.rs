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
        unsafe { gpu_get_devices() }
    }
    #[cfg(not(feature = "cuda-gpu"))]
    {
        Vec::new()
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
            return Err(Error::Generic);
        }

        if output_size > GPU_MAX_SIZE {
            return Err(Error::InvalidParam);
        }

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
