# GPU Performance Benchmarks

This directory contains performance benchmarks for GPU acceleration features.

## Running Benchmarks

### Prerequisites

1. Build ripgrep with GPU support:
```bash
cargo build --release --features cuda-gpu
```

2. Ensure you have an NVIDIA GPU with CUDA support (optional - benchmarks will show CPU-only results if not available)

### Run All Benchmarks

```bash
chmod +x benchmarks/gpu_benchmark.sh
./benchmarks/gpu_benchmark.sh
```

This will:
- Generate test files of various sizes (1MB, 100MB, 1GB)
- Run CPU and GPU benchmarks for each file
- Generate a results report in `benchmarks/gpu_benchmark_results.md`

### Expected Runtime

- With test data already generated: ~30 seconds
- First run (generating test data): ~2-5 minutes depending on storage speed

## Benchmark Results

Results are saved to `gpu_benchmark_results.md` and include:
- System configuration (GPU model, CPU, memory)
- Performance comparison table (CPU vs GPU times)
- Speedup calculations
- Analysis and recommendations

## Understanding Results

### File Size Impact

- **<10MB**: CPU is typically faster (GPU transfer overhead dominates)
- **10MB-10GB**: GPU may show 10-30% improvement
- **>10GB**: GPU should show 2-5x improvement
- **>50GB**: GPU designed for these sizes, 4-10x improvement expected

### Performance Factors

Several factors affect GPU performance:

1. **GPU Hardware**
   - Compute capability (7.0+ required, 8.0+ recommended)
   - Memory bandwidth (higher is better)
   - Available memory (limits max file size)

2. **Storage Speed**
   - NVMe: Best performance
   - SATA SSD: Good performance
   - HDD: May become bottleneck

3. **File Characteristics**
   - Compressibility (more compressed = less data to transfer)
   - Pattern frequency (more matches = more processing)

4. **System Load**
   - Other GPU tasks reduce available resources
   - CPU threads compete for memory bandwidth

## Interpreting Speedup Values

| Speedup | Interpretation |
|---------|----------------|
| < 1.0x  | GPU slower (expected for small files) |
| 1.0-1.5x | Marginal improvement |
| 1.5-3.0x | Good improvement |
| 3.0-5.0x | Excellent improvement |
| > 5.0x  | Outstanding improvement |

## Custom Benchmarks

You can create custom benchmarks by modifying the script:

```bash
# Edit gpu_benchmark.sh and add:
benchmark_file $((YOUR_SIZE)) "YOUR_NAME" "YOUR_PATTERN"
```

## Continuous Benchmarking

For CI/CD integration:

```bash
# Run benchmarks and check for regressions
./benchmarks/gpu_benchmark.sh
# Parse results and fail if performance degrades
```

## Known Limitations

1. **No GPU Available**: Benchmarks will show CPU-only results
2. **First Run Slower**: File generation takes time
3. **Compressed Files**: Benchmark uses gzip format, not GDeflate native format
4. **Pattern Simplicity**: Simple pattern matching may not stress GPU fully

## Troubleshooting

### Benchmark Fails to Run

```bash
# Check GPU support compiled in
./target/release/rg --version | grep cuda-gpu

# Verify CUDA toolkit
nvidia-smi

# Check environment
echo $CUDA_PATH
```

### Unexpected Results

- Ensure no other GPU tasks running: `nvidia-smi`
- Check storage isn't full: `df -h`
- Verify CPU isn't throttled: `lscpu | grep MHz`
- Run multiple times to account for caching

## Benchmark Data

Test data is stored in `benchmarks/testdata/` and can be safely deleted:

```bash
rm -rf benchmarks/testdata/
```

It will be regenerated on next benchmark run.

## Contributing

To add new benchmarks:

1. Add benchmark function to `gpu_benchmark.sh`
2. Document expected behavior
3. Update this README
4. Test on systems with and without GPU
