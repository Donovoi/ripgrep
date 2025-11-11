# Configuration Guide: DirectStorage/GDeflate Integration

## Overview

This document explains how to configure and use DirectStorage/GDeflate integration with ripgrep once implemented.

## Feature Flag

GDeflate support is an optional feature that must be enabled at compile time:

```bash
# Default build (without GDeflate)
cargo build --release

# Build with GDeflate support
cargo build --release --features gdeflate

# Build with all features including GDeflate
cargo build --release --all-features
```

## Cargo.toml Configuration

Add the following to your `Cargo.toml` to enable GDeflate support:

```toml
[features]
default = []
gdeflate = ["gdeflate-sys"]

[dependencies]
# Only included when gdeflate feature is enabled
gdeflate-sys = { git = "https://github.com/Donovoi/DirectStorage", 
                 subdir = "GDeflate/rust", 
                 optional = true }
```

## Runtime Configuration

### Environment Variables

```bash
# Force GDeflate decompression (even if not detected)
export RG_GDEFLATE_FORCE=1

# Set number of decompression worker threads (0 = auto)
export RG_GDEFLATE_THREADS=8

# Disable GDeflate even if compiled in
export RG_GDEFLATE_DISABLE=1

# Enable verbose GDeflate logging
export RG_GDEFLATE_DEBUG=1
```

### Command Line Flags

```bash
# Use GDeflate for .gz files (if supported)
rg --gdeflate-gz pattern file.gz

# Disable GDeflate temporarily
rg --no-gdeflate pattern file.gdz

# Show decompression statistics
rg --decompress-stats pattern file.gdz

# Set decompression thread count
rg --gdeflate-threads 4 pattern file.gdz
```

## File Format Support

### Supported Extensions

When GDeflate support is enabled:

- `.gdz` - Native GDeflate format (automatic detection)
- `.gdeflate` - Alternative extension
- `.gz` - Can optionally use GDeflate if file is in GDeflate format

### File Detection

GDeflate files are detected using:

1. **Magic number**: First 4 bytes must be `GDZ\0` (0x47 0x44 0x5A 0x00)
2. **Extension**: `.gdz` or `.gdeflate` files are checked for magic number
3. **Content inspection**: If extension is ambiguous, file header is inspected

## Performance Tuning

### Thread Configuration

The optimal number of decompression threads depends on:
- Number of CPU cores
- File size
- Compression ratio
- Concurrent search operations

Recommended settings:

```bash
# Small files (< 1MB) - use fewer threads to reduce overhead
export RG_GDEFLATE_THREADS=2

# Medium files (1-10MB) - use moderate parallelism
export RG_GDEFLATE_THREADS=4

# Large files (> 10MB) - use maximum parallelism
export RG_GDEFLATE_THREADS=0  # auto-detect

# Many small files - limit to avoid thread creation overhead
export RG_GDEFLATE_THREADS=1
```

### Memory Limits

```bash
# Set maximum decompressed size per file (in MB)
export RG_GDEFLATE_MAX_SIZE=1024  # 1GB

# Enable streaming decompression for large files
export RG_GDEFLATE_STREAMING=1

# Set chunk size for streaming (in KB)
export RG_GDEFLATE_CHUNK_SIZE=256  # 256KB chunks
```

## Platform-Specific Configuration

### Windows

```powershell
# Enable GPU acceleration (if DirectStorage available)
$env:RG_GDEFLATE_GPU=1

# Prefer CPU decompression even on Windows
$env:RG_GDEFLATE_GPU=0

# DirectStorage debug logging
$env:DSTORAGE_DEBUG=1
```

### Linux

```bash
# Use maximum CPU threads for decompression
export RG_GDEFLATE_THREADS=0

# Enable perf counters for profiling
export RG_GDEFLATE_PERF=1
```

### macOS

```bash
# Optimize for Apple Silicon
export RG_GDEFLATE_THREADS=0

# Use high-performance cores only
export RG_GDEFLATE_AFFINITY=performance
```

## Integration with Build Systems

### CI/CD Configuration

#### GitHub Actions

```yaml
name: Build with GDeflate

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          submodules: recursive  # Include DirectStorage submodule
      
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build
      
      - name: Build GDeflate library
        run: |
          cd vendor/DirectStorage/GDeflate
          cmake --preset Release
          cmake --build --preset Release
          sudo cmake --install build/Linux/Release
      
      - name: Build ripgrep with GDeflate
        run: cargo build --release --features gdeflate
      
      - name: Test
        run: cargo test --features gdeflate
```

#### Docker

```dockerfile
FROM rust:latest

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    git

# Clone DirectStorage
WORKDIR /build
RUN git clone --recurse-submodules https://github.com/Donovoi/DirectStorage.git

# Build GDeflate library
WORKDIR /build/DirectStorage/GDeflate
RUN cmake --preset Release && \
    cmake --build --preset Release && \
    cmake --install build/Linux/Release

# Build ripgrep with GDeflate
WORKDIR /build/ripgrep
COPY . .
RUN cargo build --release --features gdeflate

# Final image
FROM debian:bookworm-slim
COPY --from=builder /build/ripgrep/target/release/rg /usr/local/bin/
CMD ["rg"]
```

## Testing Configuration

### Unit Tests

```bash
# Run tests with GDeflate enabled
cargo test --features gdeflate

# Run only GDeflate-specific tests
cargo test --features gdeflate gdeflate

# Run with verbose output
cargo test --features gdeflate -- --nocapture
```

### Integration Tests

```bash
# Test GDeflate decompression
rg --features gdeflate "pattern" tests/data/sample.gdz

# Compare performance with gzip
./benchsuite/gdeflate_benchmark.sh

# Stress test with large files
rg --features gdeflate "pattern" large_corpus/*.gdz
```

## Troubleshooting

### GDeflate Library Not Found

```
Error: GDeflate library not found
```

**Solution**: Install GDeflate library or ensure it's in library path:

```bash
# Linux
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

# Windows
set PATH=C:\Program Files\GDeflate\bin;%PATH%
```

### Build Errors

```
error: linking with `cc` failed
```

**Solution**: Install required build tools:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake

# Fedora/RHEL
sudo dnf install gcc-c++ cmake

# macOS
xcode-select --install
brew install cmake
```

### Performance Issues

If GDeflate is slower than expected:

1. Check thread count: `export RG_GDEFLATE_DEBUG=1`
2. Verify file format: `file your_file.gdz`
3. Test decompression directly: `GDeflateDemo /decompress file.gdz -`
4. Compare with gzip: `time rg pattern file.gz` vs `time rg pattern file.gdz`

### Compatibility Issues

If GDeflate files don't open:

1. Verify magic number: `hexdump -C file.gdz | head -1`
2. Check compression: `GDeflateDemo /decompress file.gdz test.txt`
3. Try different extensions: `.gdz`, `.gdeflate`

## Migration Guide

### Converting Existing Compressed Files

```bash
# Batch convert .gz to .gdz
for file in *.gz; do
    gunzip -c "$file" | GDeflateDemo /compress - "${file%.gz}.gdz"
done

# Verify conversion
for file in *.gdz; do
    echo "Checking $file..."
    GDeflateDemo /decompress "$file" - | diff - <(gunzip -c "${file%.gdz}.gz")
done
```

### Updating Build Scripts

Before:
```bash
cargo build --release
```

After (with GDeflate):
```bash
# Option 1: Always enable
cargo build --release --features gdeflate

# Option 2: Conditional enable
if [ "$USE_GDEFLATE" = "1" ]; then
    cargo build --release --features gdeflate
else
    cargo build --release
fi
```

## Best Practices

### When to Use GDeflate

✅ **Good Use Cases:**
- Large compressed text files
- Searching compressed log archives
- Compressed source code repositories
- Repeated searches on the same files

❌ **Poor Use Cases:**
- Very small files (< 10KB)
- Single-use compressed files
- Files that will be decompressed anyway
- Network-mounted compressed files (latency)

### Compression Guidelines

1. **Compression Level**: Use level 6-9 for good balance
2. **Chunk Size**: Keep files under 100MB for best parallelism
3. **File Format**: Use standard 12-byte header format
4. **Naming**: Use `.gdz` extension for clarity

### Security Considerations

1. **Always validate** magic numbers before decompression
2. **Set reasonable limits** on uncompressed size
3. **Watch for decompression bombs** (extreme compression ratios)
4. **Sanitize file paths** when processing archives
5. **Run with limited privileges** when processing untrusted files

## Additional Resources

- [DirectStorage Repository](https://github.com/Donovoi/DirectStorage)
- [GDeflate Documentation](https://github.com/Donovoi/DirectStorage/tree/main/GDeflate)
- [Ripgrep User Guide](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md)
- [Performance Benchmarks](./DIRECTSTORAGE_INTEGRATION.md#performance-analysis)

## Support

For issues related to:
- **GDeflate library**: Open issue at https://github.com/Donovoi/DirectStorage/issues
- **Ripgrep integration**: Open issue at https://github.com/BurntSushi/ripgrep/issues
- **Performance questions**: Check benchmark results and tuning guide above
