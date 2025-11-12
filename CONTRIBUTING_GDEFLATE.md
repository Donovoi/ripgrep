# Contributing to GDeflate Integration

This guide helps developers contribute to the GDeflate integration for ripgrep.

## Overview

The GDeflate integration is designed to speed up compressed file searching by 3-8x through parallel decompression. The foundation is complete; implementation awaits the GDeflate Rust library.

## Current Status

See [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md) for detailed status.

**TL;DR**: Foundation complete (feature flags, docs, example), waiting for GDeflate library.

## Quick Start for Contributors

### 1. Build and Test

```bash
# Clone and build
git clone https://github.com/Donovoi/ripgrep.git
cd ripgrep
cargo build

# Run tests
cargo test

# Run example (without GDeflate)
cargo run --example gdeflate_integration

# Run example (with GDeflate feature, stub only for now)
cargo run --example gdeflate_integration --features gdeflate
```

### 2. Understand the Architecture

Read these documents in order:
1. [QUICKSTART.md](./QUICKSTART.md) - 5 min overview
2. [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md) - Current status
3. [DIRECTSTORAGE_INTEGRATION.md](./DIRECTSTORAGE_INTEGRATION.md) - Technical details
4. [examples/gdeflate_integration.rs](./examples/gdeflate_integration.rs) - Code example

### 3. Key Files to Know

| File | Purpose | Status |
|------|---------|--------|
| `Cargo.toml` | Feature flag configuration | ✅ Complete |
| `examples/gdeflate_integration.rs` | Proof-of-concept | ✅ Complete |
| `crates/cli/src/decompress.rs` | Decompression system | ⏸️ Awaiting integration |
| `tests/gdeflate_tests.rs` | Integration tests | ❌ Not created yet |
| `benchsuite/gdeflate_benchmark.sh` | Performance tests | ✅ Complete |

## How to Contribute

### Option 1: Work on Documentation

Even while waiting for the library, you can:

- Improve existing documentation
- Add more usage examples
- Create video tutorials or blog posts
- Translate documentation

### Option 2: Prepare Test Infrastructure

Help prepare for implementation:

- Create test data files
- Design test cases
- Set up CI/CD for performance testing
- Plan cross-platform testing strategy

### Option 3: Implement GDeflate Integration (When Ready)

Once the GDeflate Rust library is available:

#### Step 1: Add Dependency

Edit `Cargo.toml`:

```toml
[dependencies]
# Add when available
gdeflate = { version = "0.1", optional = true }

[features]
gdeflate = ["dep:gdeflate"]
```

#### Step 2: Implement GDeflateReader

In `crates/cli/src/decompress.rs`:

```rust
#[cfg(feature = "gdeflate")]
pub struct GDeflateReader {
    inner: gdeflate::Decompressor,
    // ... implementation details
}

#[cfg(feature = "gdeflate")]
impl GDeflateReader {
    pub fn new(file: File) -> io::Result<Self> {
        // Validate magic number
        // Initialize decompressor
        // Set up parallel workers
    }
}
```

#### Step 3: Integrate with DecompressionMatcher

Update the matching logic to detect `.gdz` files and use GDeflateReader.

#### Step 4: Add Tests

Create `tests/gdeflate_tests.rs`:

```rust
#[test]
#[cfg(feature = "gdeflate")]
fn test_gdeflate_search() {
    // Test searching .gdz files
}
```

#### Step 5: Benchmark

Run the benchmark suite:

```bash
./benchsuite/gdeflate_benchmark.sh
```

Verify 3-8x speedup is achieved.

## Development Workflow

### 1. Make Changes

```bash
# Create feature branch
git checkout -b feature/my-contribution

# Make changes
vim files...

# Test changes
cargo test
cargo clippy
```

### 2. Follow Best Practices

- **Minimal Changes**: Change only what's necessary
- **Test Everything**: Add tests for new functionality
- **Document**: Update docs to reflect changes
- **No Warnings**: Code must compile without warnings
- **Clippy Clean**: Fix all clippy suggestions

### 3. Submit PR

```bash
# Commit changes
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin feature/my-contribution

# Create PR on GitHub
```

## Testing Guidelines

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With features
cargo test --features gdeflate

# Release mode (for performance)
cargo test --release
```

### Test Structure

Tests should:
- Be isolated and independent
- Run quickly (< 1 second each)
- Have clear assertions
- Include both success and failure cases
- Test edge cases (empty files, huge files, etc.)

### Performance Testing

```bash
# Baseline performance
time rg -z "pattern" large_compressed_file.gz

# With GDeflate (once implemented)
time rg "pattern" large_compressed_file.gdz

# Automated benchmarks
./benchsuite/gdeflate_benchmark.sh
```

## Code Style

This project follows Rust standard style:

```bash
# Format code
cargo fmt

# Check style
cargo fmt -- --check

# Lint code
cargo clippy -- -D warnings
```

## Security Considerations

When implementing GDeflate support:

1. **Validate Magic Numbers**: Always check file headers
2. **Size Limits**: Prevent decompression bombs (max 1GB uncompressed)
3. **Compression Ratio**: Reject suspicious ratios (> 1000:1)
4. **Error Handling**: Gracefully handle corrupted files
5. **Memory Safety**: Use safe Rust, avoid unsafe unless necessary

Example security checks:

```rust
const MAX_UNCOMPRESSED_SIZE: usize = 1024 * 1024 * 1024; // 1GB
const MAX_COMPRESSION_RATIO: usize = 1000;

if uncompressed_size > MAX_UNCOMPRESSED_SIZE {
    return Err("File too large");
}

if uncompressed_size > compressed_size * MAX_COMPRESSION_RATIO {
    return Err("Suspicious compression ratio");
}
```

## Performance Optimization Tips

1. **Profile First**: Use `cargo flamegraph` or `perf` to find bottlenecks
2. **Benchmark Changes**: Always measure before and after
3. **Test on Real Data**: Synthetic benchmarks can be misleading
4. **Consider Trade-offs**: Speed vs memory vs code complexity
5. **Platform-Specific**: Optimize for common platforms (Linux, macOS, Windows)

## Common Issues and Solutions

### "Feature 'gdeflate' not found"

**Problem**: Trying to use gdeflate before it's implemented.

**Solution**: The feature flag exists but the actual library isn't integrated yet. Check [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md).

### "Tests fail with gdeflate enabled"

**Problem**: Feature flag enabled but implementation is stub-only.

**Solution**: Current implementation is proof-of-concept. Wait for actual library integration.

### "Performance isn't 3-8x faster"

**Problem**: Not seeing expected speedup.

**Solution**: 
- Ensure using release build (`--release`)
- Check file sizes (small files have less speedup)
- Verify GDeflate is actually being used (check with `--debug`)
- Profile to find actual bottleneck

## Resources

### Documentation
- [Rust Book](https://doc.rust-lang.org/book/)
- [ripgrep Guide](./GUIDE.md)
- [GDeflate Analysis](./DIRECTSTORAGE_INTEGRATION.md)

### Tools
- [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph) - Profiling
- [hyperfine](https://github.com/sharkdp/hyperfine) - Benchmarking
- [cargo-watch](https://github.com/watchexec/cargo-watch) - Auto-rebuild

### Community
- GitHub Issues: Report bugs and request features
- Pull Requests: Contribute code
- Discussions: Ask questions and share ideas

## Questions?

- Check [FAQ.md](./FAQ.md) for common questions
- Review [INTEGRATION_STATUS.md](./INTEGRATION_STATUS.md) for current status
- Open a GitHub issue for bugs or questions
- Submit a PR for improvements

## License

This project is dual-licensed under MIT or UNLICENSE. Contributions are accepted under the same terms.
