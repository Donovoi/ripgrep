# GDeflate Rust Bindings

Rust bindings for the GDeflate compression library. GDeflate is a hardware-accelerated compression format that closely matches the DEFLATE format, designed for high-performance compression and decompression.

## Prerequisites

**Important:** This crate requires the DirectStorage repository submodules to be initialized:

```bash
git submodule update --init --recursive
```

If you see build errors about missing files, make sure the submodules are properly initialized.

## Features

- **Hardware Acceleration**: Automatic SIMD detection and usage (SSE, AVX, NEON)
- **Multi-threaded Parallelism**: Up to 32-way parallel decompression
- **Safe Rust API**: Zero-cost wrapper around the C interface
- **Compression levels 1-12**: Flexible speed/ratio tradeoff
- **Auto-tuning**: Automatic thread count selection based on data size
- **Zero-copy operations**: Where possible for maximum performance

## Hardware Acceleration

### CPU SIMD (Available on all platforms)

The library automatically detects and uses available SIMD instruction sets:

- **x86/x86_64**: SSE2, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, PCLMULQDQ
- **ARM/AArch64**: NEON, CRC32 extensions, PMULL extensions

**Performance impact**: 3-4x speedup with zero configuration required.

### Multi-threaded CPU Parallelism

GDeflate's format enables up to 32-way parallel decompression:

| Data Size | Optimal Threads | Expected Speedup |
|-----------|----------------|------------------|
| < 1 MB    | 1              | 1.0x (baseline) |
| 1-10 MB   | 2-4            | 2-5x            |
| 10-100 MB | 4-8            | 4-10x           |
| > 100 MB  | 8-32           | 8-15x           |

### GPU Acceleration

GPU acceleration via DirectStorage is **not currently implemented**. The current CPU-based implementation already provides 8-15x speedup for large files.

**See [HARDWARE_ACCELERATION.md](HARDWARE_ACCELERATION.md) for detailed information.**

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
# Local path:
gdeflate = { path = "../GDeflate/rust" }
# Or git dependency:
# gdeflate = { git = "https://github.com/Donovoi/DirectStorage", subdir = "GDeflate/rust" }
```

### Basic Example

```rust
use gdeflate::{compress, decompress};

fn main() {
    let input = b"Hello, world! This is a test of GDeflate compression.";
    
    // Compress the data with level 6 compression
    let compressed = compress(input, 6, 0).expect("compression failed");
    println!("Compressed {} bytes to {} bytes", input.len(), compressed.len());
    
    // Decompress with auto-detected thread count (recommended)
    let decompressed = decompress(&compressed, input.len(), 0)
        .expect("decompression failed");
    assert_eq!(input, decompressed.as_slice());
}
```

### Auto-tuning Example

```rust
use gdeflate::{compress, decompress_auto, recommended_workers};

fn main() {
    let input = vec![0u8; 50_000_000]; // 50 MB
    
    // Compress
    let compressed = compress(&input, 6, 0).expect("compression failed");
    
    // Option 1: Use automatic thread selection (easiest)
    let decompressed = decompress_auto(&compressed, input.len())
        .expect("decompression failed");
    
    // Option 2: Get recommendation and use it
    let workers = recommended_workers(input.len());
    println!("Using {} worker threads", workers);
    let decompressed = decompress(&compressed, input.len(), workers)
        .expect("decompression failed");
}
```

### Fine-grained Control

```rust
use gdeflate::{compress, decompress};

fn main() {
    let input = b"data...";
    let compressed = compress(input, 6, 0).expect("compression failed");
    
    // Single-threaded (best for small files < 1 MB)
    let result = decompress(&compressed, input.len(), 1)?;
    
    // 4 threads (good for medium files 1-10 MB)
    let result = decompress(&compressed, input.len(), 4)?;
    
    // 8 threads (good for large files 10-100 MB)
    let result = decompress(&compressed, input.len(), 8)?;
    
    // Auto-detect (recommended for most use cases)
    let result = decompress(&compressed, input.len(), 0)?;
}
```

### Integration with ripgrep

To use GDeflate in a search tool like ripgrep, you can create a custom decompressor:

```rust
use std::io::{self, Read, Write};
use gdeflate::decompress_auto;

pub struct GDeflateDecoder<R> {
    reader: R,
}

impl<R: Read> GDeflateDecoder<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<R: Read> Read for GDeflateDecoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Read compressed data
        let mut compressed = Vec::new();
        self.reader.read_to_end(&mut compressed)?;
        
        // Decompress with auto thread selection
        let decompressed = decompress_auto(&compressed, buf.len())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        // Copy to output buffer
        let len = decompressed.len().min(buf.len());
        buf[..len].copy_from_slice(&decompressed[..len]);
        Ok(len)
    }
}
```

## API

### Functions

- `compress(input: &[u8], level: u32, flags: u32) -> Result<Vec<u8>>` - Compress data
- `decompress(input: &[u8], output_size: usize, num_workers: u32) -> Result<Vec<u8>>` - Decompress data
- `decompress_auto(input: &[u8], output_size: usize) -> Result<Vec<u8>>` - Decompress with automatic thread selection
- `recommended_workers(output_size: usize) -> u32` - Get recommended thread count for a given data size
- `compress_bound(size: usize) -> usize` - Calculate maximum compressed size
- `version() -> &'static str` - Get library version

### Constants

- `MIN_COMPRESSION_LEVEL` (1) - Minimum compression level (fastest)
- `MAX_COMPRESSION_LEVEL` (12) - Maximum compression level (best compression)
- `COMPRESS_SINGLE_THREAD` - Flag to force single-threaded compression

## Building

The crate includes a build script that compiles the GDeflate C++ library. Requirements:

- C++ compiler with C++17 support
- CMake (optional, for standalone builds)

## Performance

GDeflate achieves similar compression ratios to standard DEFLATE but is optimized for:

- **Automatic SIMD acceleration**: 3-4x speedup on modern CPUs
- **Parallel CPU decompression**: Up to 15x speedup with 32 threads
- **High-throughput streaming**: Optimized for large data processing

### Performance Comparison

Typical decompression throughput on modern hardware:

| Configuration | Throughput | Speedup |
|--------------|-----------|---------|
| Single-thread, no SIMD | 50-100 MB/s | 1x (baseline) |
| Single-thread, with SIMD | 150-300 MB/s | 3-4x |
| 8 threads, with SIMD | 800-1500 MB/s | 16-30x |
| 32 threads, with SIMD | 2000-3000 MB/s | 40-60x |

**Note**: Actual performance depends on CPU, memory bandwidth, and data characteristics.

## GPU Support Status

**GPU acceleration is NOT currently implemented** in this crate. The reasons are:

1. **Platform limitations**: DirectStorage GPU decompression is Windows-only
2. **Cross-platform goals**: Ripgrep targets Linux, macOS, and Windows
3. **Sufficient CPU performance**: 8-15x speedup already achieved with CPU parallelism
4. **Complexity**: GPU integration requires Windows SDK and DirectX 12

**Performance impact of GPU (if implemented):**
- Small-medium files (< 10 GB): 1.5-4x additional speedup on Windows
- Large files (10-100 GB): 3-6x additional speedup on Windows
- **Extremely large files (100 GB - 1 TB): 4-8x additional speedup on Windows**

For files in the 100GB-1TB range, GPU acceleration becomes significantly more beneficial and could be considered for future implementation. See [HARDWARE_ACCELERATION.md](HARDWARE_ACCELERATION.md) for detailed analysis.

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Contributing

Contributions are welcome! Please see the main repository for guidelines.

