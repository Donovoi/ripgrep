//! Rust bindings for GDeflate compression library
//!
//! GDeflate is a GPU-optimized compression format that closely matches the DEFLATE format.
//! This crate provides safe Rust bindings to the GDeflate C API, making it easy to integrate
//! high-performance compression into Rust applications.
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
//! // Decompress the data
//! let decompressed = decompress(&compressed, input.len(), 0).expect("decompression failed");
//! assert_eq!(input, decompressed.as_slice());
//! ```

use std::os::raw::{c_uint, c_int};

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
/// Compressed data as a Vec<u8>
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
    if level < MIN_COMPRESSION_LEVEL || level > MAX_COMPRESSION_LEVEL {
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
/// * `num_workers` - Number of worker threads to use (0 for default)
///
/// # Returns
///
/// Decompressed data as a Vec<u8>
///
/// # Errors
///
/// Returns an error if decompression fails or parameters are invalid
///
/// # Example
///
/// ```no_run
/// use gdeflate::{compress, decompress};
///
/// let input = b"Hello, world!";
/// let compressed = compress(input, 6, 0).expect("compression failed");
/// let decompressed = decompress(&compressed, input.len(), 0).expect("decompression failed");
/// assert_eq!(input, decompressed.as_slice());
/// ```
pub fn decompress(input: &[u8], output_size: usize, num_workers: u32) -> Result<Vec<u8>> {
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
        assert!(compressed.len() > 0);
        println!("Compressed {} bytes to {} bytes", input.len(), compressed.len());
        
        // Decompress
        let decompressed = decompress(&compressed, input.len(), 0).expect("decompression failed");
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
            let compressed = compress(&input_bytes, level, 0)
                .expect(&format!("compression failed at level {}", level));
            let decompressed = decompress(&compressed, input_bytes.len(), 0)
                .expect(&format!("decompression failed at level {}", level));
            assert_eq!(input_bytes, decompressed);
        }
    }
}
