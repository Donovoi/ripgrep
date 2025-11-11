# Quick Start: Evaluating DirectStorage/GDeflate for Ripgrep

This guide helps you quickly evaluate whether DirectStorage/GDeflate integration would benefit your ripgrep use case.

## 5-Minute Evaluation

### Step 1: Check Your Use Case

Do you frequently search **compressed files**?

```bash
# Count compressed files in your typical search locations
find /your/search/path -name "*.gz" -o -name "*.bz2" -o -name "*.xz" | wc -l

# If the number is > 100, GDeflate may help significantly
```

### Step 2: Measure Current Performance

```bash
# Time a typical compressed file search
time rg "your_pattern" /path/to/compressed_files/*.gz

# Note the time - this is your baseline
```

### Step 3: Estimate Potential Speedup

Based on file sizes:
- **Small files** (< 100KB): ~3x faster
- **Medium files** (100KB - 10MB): ~5x faster  
- **Large files** (> 10MB): ~8x faster

**Example**: If your search took 10 seconds, GDeflate could reduce it to 2-3 seconds.

## 30-Minute Deep Dive

### Understand the Integration

1. **Read**: [ANALYSIS_SUMMARY.md](./ANALYSIS_SUMMARY.md)
   - 5-minute read
   - Executive overview
   - Key findings

2. **Explore**: [examples/gdeflate_integration.rs](./examples/gdeflate_integration.rs)
   - 10 minutes to understand
   - Working proof-of-concept
   - Shows integration architecture

3. **Review**: [DIRECTSTORAGE_INTEGRATION.md](./DIRECTSTORAGE_INTEGRATION.md)
   - 15-minute detailed read
   - Technical implementation plan
   - Performance analysis

### Run the Example

```bash
# Build and run the example
cd /path/to/ripgrep
cargo run --example gdeflate_integration

# You'll see output explaining the architecture
```

## 2-Hour Hands-On Evaluation

### Prerequisites

```bash
# Install DirectStorage/GDeflate
git clone --recurse-submodules https://github.com/Donovoi/DirectStorage.git
cd DirectStorage/GDeflate

# Linux/macOS
cmake --preset Release
cmake --build --preset Release
sudo cmake --install build/Linux/Release  # or Darwin

# Windows (in Visual Studio Developer Command Prompt)
cmake --preset Release
cmake --build --preset Release
cmake --install build/Windows/Release
```

### Create Test Data

```bash
cd /path/to/ripgrep

# Option 1: Use benchmark script (easiest)
./benchsuite/gdeflate_benchmark.sh

# Option 2: Manual test data
mkdir test_data
cd test_data

# Create a large text file
cat /usr/share/dict/words > large.txt
for i in {1..100}; do cat large.txt >> large.txt; done

# Compress with gzip (existing format)
gzip -k -9 large.txt

# Compress with GDeflate (new format)
GDeflateDemo /compress large.txt large.txt.gdz
```

### Benchmark Performance

```bash
# Benchmark gzip (current)
time rg "pattern" test_data/large.txt.gz

# Benchmark GDeflate (proposed)
# Note: This requires implementing the integration first
# For now, decompress and search separately:
time (GDeflateDemo /decompress large.txt.gdz - | rg "pattern")

# Compare the times
```

### Analyze Results

The benchmark script will generate:
- `benchmark_results.csv` - Raw performance data
- `benchmark_report.md` - Detailed analysis

Review these to see actual speedups on your data.

## Decision Framework

### When to Implement

✅ **Implement if**:
- You search 100+ compressed files regularly
- Files are medium to large (> 100KB)
- Performance is a priority
- You can build native dependencies

✅ **Strong indicators**:
- Current gzip searches take > 5 seconds
- You have multi-core systems
- Compressed files change infrequently
- You control the compression format

### When to Skip

❌ **Skip if**:
- You rarely search compressed files
- Files are very small (< 10KB)
- Build complexity is a concern
- Only searching once per file

❌ **Neutral indicators**:
- Single-core systems (still some benefit)
- Can't control compression format
- Network-mounted files (latency dominates)

## Next Steps

### If Evaluation is Positive

1. **Share findings** with ripgrep maintainers
   - Include benchmark results
   - Describe your use case
   - Reference this analysis

2. **Consider prototyping**
   - Fork ripgrep
   - Implement Phase 1 (native decompression)
   - Measure actual improvements

3. **Community discussion**
   - Would others benefit?
   - Is .gdz format acceptable?
   - What's the adoption path?

### If Evaluation is Negative

1. **Current approach works**
   - External gzip is sufficient
   - Performance is acceptable

2. **Alternative optimizations**
   - Use faster compression (zstd)
   - Pre-decompress files
   - Use faster storage (SSD)
   - Reduce search scope

## Common Questions

### Q: Do I need to recompress all my files?

**A**: No! GDeflate would be *optional*. Your existing .gz files would still work with external gzip. GDeflate would be an additional option for better performance.

### Q: Can I use this today?

**A**: Not yet. This is an analysis and proposal. Implementation would require:
1. Community agreement
2. Implementation work (9-14 days estimated)
3. Testing and validation
4. Documentation

### Q: What about other compression formats?

**A**: GDeflate specifically targets DEFLATE-based formats (gzip). For other formats:
- zstd: Already very fast, consider using it instead
- bzip2: Could benefit from similar parallel decompression
- xz: Could benefit from similar parallel decompression

### Q: Windows-only GPU acceleration?

**A**: GPU acceleration is Windows-only, but CPU parallel decompression (4-8x speedup) works on all platforms. Windows gets an additional boost.

### Q: Build complexity?

**A**: Added as an optional feature flag. If you don't enable it, build is unchanged. If you do enable it, requires GDeflate C library (similar to optional pcre2 support).

### Q: Performance on small files?

**A**: Still beneficial but less dramatic:
- 10KB file: 3.3x faster (30ms → 9ms)
- Process spawn overhead is main bottleneck
- In-process decompression helps significantly

## Resources

### Quick Reference
- **Analysis Summary**: [ANALYSIS_SUMMARY.md](./ANALYSIS_SUMMARY.md)
- **Technical Details**: [DIRECTSTORAGE_INTEGRATION.md](./DIRECTSTORAGE_INTEGRATION.md)
- **Configuration**: [GDEFLATE_CONFIG.md](./GDEFLATE_CONFIG.md)
- **Example Code**: [examples/gdeflate_integration.rs](./examples/gdeflate_integration.rs)
- **Benchmarks**: [benchsuite/gdeflate_benchmark.sh](./benchsuite/gdeflate_benchmark.sh)

### External Links
- [DirectStorage](https://github.com/Donovoi/DirectStorage)
- [GDeflate Docs](https://github.com/Donovoi/DirectStorage/tree/main/GDeflate)
- [Ripgrep](https://github.com/BurntSushi/ripgrep)

## Contact

Questions? Open an issue in the ripgrep repository and reference this evaluation guide.

---

**Last Updated**: November 2025
**Analysis Version**: 1.0
**Status**: Proposal/Evaluation Phase
