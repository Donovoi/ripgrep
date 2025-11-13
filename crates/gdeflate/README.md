# GDeflate Rust Bindings

Rust bindings for the GDeflate compression library. GDeflate is a GPU-optimized compression format that closely matches the DEFLATE format, designed for high-performance compression and decompression.

## Prerequisites

**Important:** This crate requires the DirectStorage repository submodules to be initialized:

```bash
git submodule update --init --recursive
```

If you see build errors about missing files, make sure the submodules are properly initialized.

## Features

- Safe Rust API wrapping the C interface
- Support for compression levels 1-12
- Multi-threaded decompression support
- Zero-copy operations where possible
- Compatible with standard DEFLATE compression ratios

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
# Local path:
gdeflate = { path = "../GDeflate/rust" }
# Or git dependency:
# gdeflate = { git = "https://github.com/Donovoi/DirectStorage", subdir = "GDeflate/rust" }
```

### Example

```rust
use gdeflate::{compress, decompress};

fn main() {
    let input = b"Hello, world! This is a test of GDeflate compression.";
    
    // Compress the data with level 6 compression
    let compressed = compress(input, 6, 0).expect("compression failed");
    println!("Compressed {} bytes to {} bytes", input.len(), compressed.len());
    
    // Decompress the data
    let decompressed = decompress(&compressed, input.len(), 0).expect("decompression failed");
    assert_eq!(input, decompressed.as_slice());
}
```

### Integration with ripgrep

To use GDeflate in a search tool like ripgrep, you can create a custom decompressor:

```rust
use std::io::{self, Read, Write};
use gdeflate::decompress;

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
        
        // Decompress
        let decompressed = decompress(&compressed, buf.len(), 0)
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

- GPU decompression (when using DirectStorage API on Windows)
- Parallel CPU decompression
- High-throughput streaming applications

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Contributing

Contributions are welcome! Please see the main repository for guidelines.
