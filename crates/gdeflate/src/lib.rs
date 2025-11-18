//! Rust bindings for GDeflate compression library
//!
//! GDeflate is a hardware-accelerated compression format that closely matches the DEFLATE format.
//! This crate provides safe Rust bindings to the GDeflate C API, making it easy to integrate
//! high-performance compression into Rust applications.
//!
//! # Hardware Acceleration
//!
//! GDeflate leverages multiple levels of hardware acceleration:
//!
//! ## CPU SIMD (Available on all platforms)
//!
//! The library automatically detects and uses available SIMD instruction sets:
//! - **x86/x86_64**: SSE2, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, PCLMULQDQ
//! - **ARM/AArch64**: NEON, CRC32 extensions, PMULL extensions
//!
//! SIMD provides 3-4x speedup with zero configuration required.
//!
//! ## Multi-threaded CPU Parallelism (Available on all platforms)
//!
//! GDeflate's format enables up to 32-way parallel decompression. The `num_workers`
//! parameter controls thread count:
//! - `0` = Auto-detect optimal thread count (recommended)
//! - `1` = Single-threaded (best for small files < 1 MB)
//! - `2-32` = Specific thread count
//!
//! Expected speedup:
//! - Small files (< 1 MB): 1-2x
//! - Medium files (1-100 MB): 3-10x
//! - Large files (> 100 MB): 8-15x
//!
//! ## GPU Acceleration (NVIDIA CUDA - Optional)
//!
//! NVIDIA GPU acceleration is available for extremely large files (50GB+) when the
//! `cuda-gpu` feature is enabled. GPU acceleration provides:
//! - **50-100GB files**: 4-8x faster than CPU multi-threading
//! - **100-500GB files**: 6-10x faster than CPU
//! - **500GB+ files**: 8-15x faster than CPU
//!
//! GPU acceleration requires:
//! - NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
//! - CUDA Toolkit 11.0 or later
//! - Enabled at compile time with `--features cuda-gpu`
//!
//! The implementation automatically falls back to CPU when GPU is unavailable.
//!
//! See [HARDWARE_ACCELERATION.md](../HARDWARE_ACCELERATION.md) for detailed information.
//!
//! # Example
//!
//! ```no_run
//! use gdeflate::{compress, decompress, compress_bound};
//!
//! let input = b"Hello, world! This is a test of GDeflate compression.";
//!
//! // Compress the data
//! let compressed = compress(input, 6, 0).expect("compression failed");
//! println!("Compressed {} bytes to {} bytes", input.len(), compressed.len());
//!
//! // Decompress with auto-detected thread count (recommended)
//! let decompressed = decompress(&compressed, input.len(), 0).expect("decompression failed");
//! assert_eq!(input, decompressed.as_slice());
//!
//! // Or use specific thread count for fine control
//! let decompressed = decompress(&compressed, input.len(), 8).expect("decompression failed");
//! assert_eq!(input, decompressed.as_slice());
//! ```

use std::os::raw::{c_int, c_uint};

// GPU acceleration module (NVIDIA CUDA support)
#[cfg(feature = "cuda-gpu")]
pub mod gpu;

// Re-export GPU module for convenience when feature is enabled
#[cfg(feature = "cuda-gpu")]
pub use gpu::{
    decompress_auto as decompress_gpu_auto, decompress_with_gpu,
    estimate_literal_search_config, estimate_literal_search_config_with_hint,
    get_gpu_devices, is_gpu_available, should_use_gpu, substring_contains,
    GpuInfo, LiteralSearchConfig, LiteralSearchInputHint,
    LiteralSearchInputKind, GPU_MAX_SIZE, GPU_SIZE_THRESHOLD,
};

/// Minimum compression level
pub const MIN_COMPRESSION_LEVEL: u32 = 1;

/// Maximum compression level
pub const MAX_COMPRESSION_LEVEL: u32 = 12;

/// Force compression using a single thread
pub const COMPRESS_SINGLE_THREAD: u32 = 0x200;

/// Error type for GDeflate operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Generic error
    Generic,
    /// Invalid parameter
    InvalidParam,
    /// Output buffer too small
    BufferTooSmall,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Generic => write!(f, "GDeflate operation failed"),
            Error::InvalidParam => write!(f, "Invalid parameter"),
            Error::BufferTooSmall => write!(f, "Output buffer too small"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type for GDeflate operations
pub type Result<T> = std::result::Result<T, Error>;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GDeflateResult {
    Success = 0,
    Error = -1,
    InvalidParam = -2,
    BufferTooSmall = -3,
}

impl From<c_int> for GDeflateResult {
    fn from(value: c_int) -> Self {
        match value {
            0 => GDeflateResult::Success,
            -2 => GDeflateResult::InvalidParam,
            -3 => GDeflateResult::BufferTooSmall,
            _ => GDeflateResult::Error,
        }
    }
}

impl From<GDeflateResult> for Result<()> {
    fn from(result: GDeflateResult) -> Self {
        match result {
            GDeflateResult::Success => Ok(()),
            GDeflateResult::InvalidParam => Err(Error::InvalidParam),
            GDeflateResult::BufferTooSmall => Err(Error::BufferTooSmall),
            GDeflateResult::Error => Err(Error::Generic),
        }
    }
}

extern "C" {
    fn gdeflate_compress_bound(size: usize) -> usize;

    fn gdeflate_compress(
        output: *mut u8,
        output_size: *mut usize,
        input: *const u8,
        input_size: usize,
        level: c_uint,
        flags: c_uint,
    ) -> c_int;

    fn gdeflate_decompress(
        output: *mut u8,
        output_size: usize,
        input: *const u8,
        input_size: usize,
        num_workers: c_uint,
    ) -> c_int;

    fn gdeflate_version() -> *const std::os::raw::c_char;
}

/// Calculate the maximum size of compressed data given input size
///
/// # Arguments
///
/// * `size` - Size of input data in bytes
///
/// # Returns
///
/// Maximum possible size of compressed data in bytes
pub fn compress_bound(size: usize) -> usize {
    unsafe { gdeflate_compress_bound(size) }
}

/// Compress data using GDeflate format
///
/// # Arguments
///
/// * `input` - Input data to compress
/// * `level` - Compression level (1-12, where 1 is fastest and 12 is best compression)
/// * `flags` - Compression flags (see COMPRESS_* constants)
///
/// # Returns
///
/// Compressed data as a `Vec<u8>`
///
/// # Errors
///
/// Returns an error if compression fails or parameters are invalid
///
/// # Example
///
/// ```no_run
/// use gdeflate::compress;
///
/// let input = b"Hello, world!";
/// let compressed = compress(input, 6, 0).expect("compression failed");
/// ```
pub fn compress(input: &[u8], level: u32, flags: u32) -> Result<Vec<u8>> {
    if !(MIN_COMPRESSION_LEVEL..=MAX_COMPRESSION_LEVEL).contains(&level) {
        return Err(Error::InvalidParam);
    }

    let max_size = compress_bound(input.len());
    let mut output = vec![0u8; max_size];
    let mut output_size = max_size;

    let result = unsafe {
        gdeflate_compress(
            output.as_mut_ptr(),
            &mut output_size,
            input.as_ptr(),
            input.len(),
            level,
            flags,
        )
    };

    GDeflateResult::from(result).into_result()?;

    output.truncate(output_size);
    Ok(output)
}

/// Decompress data from GDeflate format
///
/// # Arguments
///
/// * `input` - Compressed input data
/// * `output_size` - Expected size of decompressed data in bytes
/// * `num_workers` - Number of worker threads to use:
///   - `0` = Auto-detect optimal thread count (recommended)
///   - `1` = Single-threaded (best for small files < 1 MB)
///   - `2-32` = Use specific number of threads
///
/// # Returns
///
/// Decompressed data as a `Vec<u8>`
///
/// # Errors
///
/// Returns an error if decompression fails or parameters are invalid
///
/// # Performance Guidelines
///
/// - **Small files (< 1 MB)**: Use `num_workers = 1` to avoid thread overhead
/// - **Medium files (1-100 MB)**: Use `num_workers = 4-8` for good balance
/// - **Large files (> 100 MB)**: Use `num_workers = 0` for auto-detection
///
/// The auto-detection mode (0) provides optimal performance for most use cases.
///
/// # Example
///
/// ```no_run
/// use gdeflate::{compress, decompress};
///
/// let input = b"Hello, world!";
/// let compressed = compress(input, 6, 0).expect("compression failed");
///
/// // Auto-detect optimal thread count (recommended)
/// let decompressed = decompress(&compressed, input.len(), 0).expect("decompression failed");
/// assert_eq!(input, decompressed.as_slice());
///
/// // Or use specific thread count
/// let decompressed = decompress(&compressed, input.len(), 8).expect("decompression failed");
/// assert_eq!(input, decompressed.as_slice());
/// ```
pub fn decompress(
    input: &[u8],
    output_size: usize,
    num_workers: u32,
) -> Result<Vec<u8>> {
    let mut output = vec![0u8; output_size];

    let result = unsafe {
        gdeflate_decompress(
            output.as_mut_ptr(),
            output_size,
            input.as_ptr(),
            input.len(),
            num_workers,
        )
    };

    GDeflateResult::from(result).into_result()?;

    Ok(output)
}

/// Get version string of GDeflate library
///
/// # Returns
///
/// Version string (e.g., "1.0.0")
pub fn version() -> &'static str {
    unsafe {
        let version_ptr = gdeflate_version();
        let version_cstr = std::ffi::CStr::from_ptr(version_ptr);
        version_cstr.to_str().unwrap_or("unknown")
    }
}

/// Recommend optimal number of worker threads for decompression
///
/// This function provides a heuristic for selecting the optimal number of
/// worker threads based on the uncompressed data size and system capabilities.
///
/// # Arguments
///
/// * `output_size` - Expected size of decompressed data in bytes
///
/// # Returns
///
/// Recommended number of worker threads (1-32)
///
/// # Guidelines
///
/// - Files < 1 MB: Returns 1 (single-threaded)
/// - Files 1-10 MB: Returns 2-4 threads
/// - Files 10-100 MB: Returns 4-8 threads
/// - Files > 100 MB: Returns 8-32 threads (capped by system CPU count)
///
/// # Example
///
/// ```no_run
/// use gdeflate::{compress, decompress, recommended_workers};
///
/// let input = vec![0u8; 50_000_000]; // 50 MB
/// let compressed = compress(&input, 6, 0).expect("compression failed");
///
/// // Get recommendation
/// let workers = recommended_workers(input.len());
/// println!("Recommended workers: {}", workers);
///
/// // Use recommendation
/// let decompressed = decompress(&compressed, input.len(), workers)
///     .expect("decompression failed");
/// ```
pub fn recommended_workers(output_size: usize) -> u32 {
    // Get available CPU parallelism
    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(8); // Default to 8 if detection fails

    // Size-based heuristics (in bytes)
    const SIZE_1MB: usize = 1024 * 1024;
    const SIZE_10MB: usize = 10 * SIZE_1MB;
    const SIZE_100MB: usize = 100 * SIZE_1MB;

    let recommended = if output_size < SIZE_1MB {
        // Small files: single-threaded to avoid overhead
        1
    } else if output_size < SIZE_10MB {
        // Medium-small files: limited parallelism
        std::cmp::min(4, cpu_count)
    } else if output_size < SIZE_100MB {
        // Medium-large files: moderate parallelism
        std::cmp::min(8, cpu_count)
    } else {
        // Large files: maximum parallelism
        std::cmp::min(32, cpu_count)
    };

    std::cmp::max(1, recommended)
}

/// Decompress data from GDeflate format with automatic thread selection
///
/// This is a convenience wrapper around `decompress` that automatically
/// selects the optimal number of worker threads based on the data size.
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
/// Returns an error if decompression fails or parameters are invalid
///
/// # Example
///
/// ```no_run
/// use gdeflate::{compress, decompress_auto};
///
/// let input = b"Hello, world!";
/// let compressed = compress(input, 6, 0).expect("compression failed");
/// let decompressed = decompress_auto(&compressed, input.len())
///     .expect("decompression failed");
/// assert_eq!(input, decompressed.as_slice());
/// ```
pub fn decompress_auto(input: &[u8], output_size: usize) -> Result<Vec<u8>> {
    let workers = recommended_workers(output_size);
    decompress(input, output_size, workers)
}

impl GDeflateResult {
    fn into_result(self) -> Result<()> {
        self.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
        assert_eq!(ver, "1.0.0");
    }

    #[test]
    fn test_compress_bound() {
        let size = compress_bound(1024);
        assert!(size >= 1024);
    }

    #[test]
    fn test_compress_decompress() {
        let input = b"Hello, world! This is a test of GDeflate compression. \
                      It should compress this text and then decompress it back.";

        // Compress
        let compressed = compress(input, 6, 0).expect("compression failed");
        assert!(!compressed.is_empty());
        println!(
            "Compressed {} bytes to {} bytes",
            input.len(),
            compressed.len()
        );

        // Decompress
        let decompressed = decompress(&compressed, input.len(), 0)
            .expect("decompression failed");
        assert_eq!(input, decompressed.as_slice());
    }

    #[test]
    fn test_invalid_compression_level() {
        let input = b"test";

        // Test level too low
        assert!(compress(input, 0, 0).is_err());

        // Test level too high
        assert!(compress(input, 13, 0).is_err());
    }

    #[test]
    fn test_compression_levels() {
        let input_text = "The quick brown fox jumps over the lazy dog. ";
        let mut input_bytes = Vec::new();
        for _ in 0..10 {
            input_bytes.extend_from_slice(input_text.as_bytes());
        }

        for level in MIN_COMPRESSION_LEVEL..=MAX_COMPRESSION_LEVEL {
            let compressed =
                compress(&input_bytes, level, 0).unwrap_or_else(|_| {
                    panic!("compression failed at level {}", level)
                });
            let decompressed = decompress(&compressed, input_bytes.len(), 0)
                .unwrap_or_else(|_| {
                    panic!("decompression failed at level {}", level)
                });
            assert_eq!(input_bytes, decompressed);
        }
    }

    #[test]
    fn test_recommended_workers() {
        // Small file - should recommend 1 thread
        assert_eq!(recommended_workers(500_000), 1); // 500 KB

        // Medium-small file - should recommend limited parallelism
        let workers = recommended_workers(5_000_000); // 5 MB
        assert!((1..=4).contains(&workers));

        // Medium-large file - should recommend moderate parallelism
        let workers = recommended_workers(50_000_000); // 50 MB
        assert!((1..=8).contains(&workers));

        // Large file - should recommend maximum available parallelism
        // (capped by system CPU count)
        let workers = recommended_workers(500_000_000); // 500 MB
        assert!((1..=32).contains(&workers));

        // Verify it scales with available CPUs
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);
        assert!(workers <= cpu_count.min(32));
    }

    #[test]
    fn test_decompress_auto() {
        let input_text = "The quick brown fox jumps over the lazy dog. ";
        let mut input_bytes = Vec::new();
        for _ in 0..100 {
            input_bytes.extend_from_slice(input_text.as_bytes());
        }

        let compressed =
            compress(&input_bytes, 6, 0).expect("compression failed");
        let decompressed = decompress_auto(&compressed, input_bytes.len())
            .expect("auto decompression failed");
        assert_eq!(input_bytes, decompressed);
    }

    #[test]
    fn test_multi_threaded_decompression() {
        // Create larger test data
        let input_text =
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
        let mut input_bytes = Vec::new();
        for _ in 0..1000 {
            input_bytes.extend_from_slice(input_text.as_bytes());
        }

        let compressed =
            compress(&input_bytes, 6, 0).expect("compression failed");

        // Test with different thread counts
        for num_workers in [1, 2, 4, 8] {
            let decompressed =
                decompress(&compressed, input_bytes.len(), num_workers)
                    .unwrap_or_else(|_| {
                        panic!(
                            "decompression failed with {} workers",
                            num_workers
                        )
                    });
            assert_eq!(
                input_bytes, decompressed,
                "Data mismatch with {} workers",
                num_workers
            );
        }
    }

    #[test]
    fn test_single_vs_multi_thread() {
        // Create test data large enough to benefit from parallelism
        let mut input_bytes = vec![0u8; 100_000];
        for (i, byte) in input_bytes.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        let compressed =
            compress(&input_bytes, 6, 0).expect("compression failed");

        // Single-threaded
        let single = decompress(&compressed, input_bytes.len(), 1)
            .expect("single-threaded decompression failed");

        // Multi-threaded
        let multi = decompress(&compressed, input_bytes.len(), 4)
            .expect("multi-threaded decompression failed");

        // Results should be identical
        assert_eq!(single, multi);
        assert_eq!(input_bytes, single);
    }
}
