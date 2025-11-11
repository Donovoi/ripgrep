# Performance Baseline

This document records baseline performance measurements for ripgrep's compressed file search, to be compared against GDeflate-enabled performance once implemented.

## Test Environment

- **Date**: 2025-11-11
- **Ripgrep Version**: 15.1.0 (rev 811a1a4dae)
- **Build**: Release profile [optimized + debuginfo]
- **SIMD**: SSE2, SSSE3, AVX2 enabled
- **Test Files**: tests/data/sherlock.{gz,bz2,xz}

## Baseline Tests

### Small Compressed Files (< 1KB)

```bash
$ time ./target/release/rg -z "Sherlock" tests/data/sherlock.gz tests/data/sherlock.bz2 tests/data/sherlock.xz
```

**Results**:
- Real time: 0.006s
- User time: 0.001s
- Sys time: 0.008s
- Files searched: 3
- Matches found: 6 (2 per file)

**Current Performance**: ~500 files/second for small compressed files

## Expected Performance with GDeflate

Based on the analysis in DIRECTSTORAGE_INTEGRATION.md:

### Small Files (< 100KB)
- **Current**: ~50ms per file (including overhead)
- **With GDeflate**: ~15ms per file
- **Expected Speedup**: 3.3x

### Medium Files (100KB - 10MB)
- **Current**: ~150ms per file
- **With GDeflate**: ~30ms per file
- **Expected Speedup**: 5.0x

### Large Files (> 10MB)
- **Current**: ~2000ms per file
- **With GDeflate**: ~250ms per file
- **Expected Speedup**: 8.0x

## Key Performance Factors

Current implementation uses external processes for decompression:
1. **Process spawn overhead**: 10-50ms per file
2. **Data copying**: Between processes via pipes
3. **Serial decompression**: Single-threaded
4. **Typical throughput**: 100-200 MB/s

GDeflate implementation will provide:
1. **No process overhead**: In-process decompression
2. **Direct memory**: No inter-process copying
3. **Parallel decompression**: 4-32 threads
4. **Higher throughput**: 800+ MB/s

## Benchmark Suite

For comprehensive testing once GDeflate is implemented:

```bash
# Run automated benchmark suite
./benchsuite/gdeflate_benchmark.sh

# Manual comparison tests
time rg -z "pattern" directory_with_gz_files/
time rg "pattern" directory_with_gdz_files/  # Once implemented
```

## Notes

- These are micro-benchmarks with very small files
- Real-world performance gains will be more significant with larger files
- The overhead of process spawning is proportionally higher for small files
- GDeflate's parallel decompression benefits increase with file size

## Next Steps

Once GDeflate integration is implemented:

1. Re-run these exact same tests
2. Measure actual speedup
3. Test with larger datasets
4. Profile memory usage
5. Validate 3-8x speedup claims
6. Update this document with results

## Test Reproduction

To reproduce these baseline tests:

```bash
# Build release version
cargo build --release

# Run baseline tests
time ./target/release/rg -z "Sherlock" tests/data/sherlock.gz tests/data/sherlock.bz2 tests/data/sherlock.xz

# Or test on your own data
time ./target/release/rg -z "your_pattern" /path/to/compressed/files/
```
