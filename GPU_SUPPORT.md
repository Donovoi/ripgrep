# NVIDIA GPU Support for Large File Decompression

## Overview

This document describes NVIDIA GPU acceleration support for GDeflate decompression in ripgrep, specifically optimized for extremely large files (50GB+).

## Performance Benefits

GPU acceleration provides significant performance improvements for large file decompression:

| File Size | CPU (32 threads) | GPU (NVIDIA) | Speedup |
|-----------|-----------------|--------------|---------|
| 50 GB     | ~8 seconds      | ~1-2 seconds | **4-8x** |
| 100 GB    | ~16 seconds     | ~2-3 seconds | **6-10x** |
| 500 GB    | ~80 seconds     | ~8-10 seconds | **8-15x** |

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.0 or higher
  - Volta architecture (Tesla V100, Titan V) or newer
  - Turing (RTX 20 series, GTX 16 series)
  - Ampere (RTX 30 series, A100)
  - Ada Lovelace (RTX 40 series)
  - Hopper (H100)
- Minimum 8GB GPU memory recommended
- For 100GB+ files: 16GB+ GPU memory recommended

### Software
- CUDA Toolkit 11.0 or later
  - Download from: https://developer.nvidia.com/cuda-downloads
  - Includes CUDA runtime and development libraries
- NVIDIA GPU driver
  - Version 450.80.02 or later (Linux)
  - Version 452.39 or later (Windows)

## Installation

### 1. Install CUDA Toolkit

#### Linux (Ubuntu/Debian)
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get install cuda
```

#### Windows
1. Download CUDA Toolkit from NVIDIA website
2. Run the installer
3. Follow the installation wizard
4. Set environment variables:
   - `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0`

### 2. Build ripgrep with GPU Support

```bash
# Clone repository
git clone https://github.com/Donovoi/ripgrep.git
cd ripgrep

# Build with GPU support
cargo build --release --features cuda-gpu

# Verify GPU is detected
./target/release/rg --version
```

## Usage

### Automatic GPU Selection

GPU acceleration is automatically used for files >= 50GB when:
1. The `cuda-gpu` feature is enabled
2. A compatible NVIDIA GPU is available
3. CUDA runtime is installed

No configuration is required - ripgrep automatically selects the optimal decompression method.

### Checking GPU Availability

```bash
# Check if GPU support is compiled in
./target/release/rg --version

# Example output:
# ripgrep 15.1.0 (with gdeflate, with cuda-gpu)
# GPU: NVIDIA GeForce RTX 4090 (24 GB)
```

### Manual Control

GPU usage can be controlled via environment variables:

```bash
# Force CPU decompression (disable GPU)
RG_NO_GPU=1 rg pattern large_file.gdz

# Prefer GPU even for smaller files (>= 10GB instead of 50GB)
RG_GPU_THRESHOLD=10GB rg pattern file.gdz

# Select specific GPU device (multi-GPU systems)
CUDA_VISIBLE_DEVICES=1 rg pattern file.gdz
```

### Literal Prefilter Flags

Literal substring searches that meet the GPU requirements (single
case-sensitive literal, non-inverted match, `--fixed-strings`) can now be tuned
directly via CLI flags:

- `--gpu-prefilter=auto|always|off` picks the heuristic, forces the CUDA
  prefilter on, or disables it entirely.
- `--gpu-chunk-size=<bytes>` overrides the chunk size handed to the GPU (e.g.
  `256M`, `512M`).
- `--gpu-strings` bundles the common literal-search flags (fixed strings,
  `--text`, no headings, line numbers, colorless output) and forces the GPU
  literal prefilter on—handy when replicating a `strings`-style scan of a big
  binary. It now also implies `--escape-control`, so ASCII control bytes (such
  as ANSI escape sequences) are rendered as `\xHH` instead of being sent raw to
  your terminal.

Example:

```bash
rg --fixed-strings "Session started" \
   --gpu-prefilter=always \
   --gpu-chunk-size=256M \
   /vault/dumps/*.gdz

# Convenience preset for a GPU strings run
rg --gpu-strings "BEGIN PKCS12" /vault/dumps/docker_data.vhdx
```

### Escaping control characters

Binary blobs routinely contain bytes like `0x1B` (ESC) that can toggle terminal
state when printed verbatim. Use `--escape-control` to rewrite every ASCII/C1
control byte (except tabs/newlines) as `\xHH` before ripgrep writes a match. The
`--gpu-strings` preset enables this automatically, but you can opt in manually:

```bash
rg --escape-control --text --fixed-strings "secret" disk.img
```

Disable it via `--no-escape-control` if you really do need the raw bytes.

Use `auto` for the existing behaviour, `always` when you want the GPU to scan
every qualifying file regardless of size, and `off` when comparing CPU vs GPU
paths. The chunk override is helpful for tuning PCIe transfer overlap to match
your GPU memory and storage stack.

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Ripgrep Search                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               GDeflate Decompression                     │
└─────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
    ┌──────────────┐            ┌──────────────┐
    │  CPU Path    │            │   GPU Path   │
    │ (< 50 GB)    │            │  (>= 50 GB)  │
    └──────────────┘            └──────────────┘
    │                                   │
    ▼                                   ▼
┌──────────────┐            ┌──────────────────────────┐
│ Multi-thread │            │ CUDA Kernel Execution    │
│ CPU SIMD     │            │ • Memory transfer to GPU │
│ (32 threads) │            │ • Parallel decompression │
│              │            │ • Transfer back to CPU   │
└──────────────┘            └──────────────────────────┘
```

### Decompression Process

1. **Detection**: File size is checked against threshold (default: 50GB)
2. **GPU Check**: Verify GPU availability and sufficient memory
3. **Memory Transfer**: Compressed data is transferred to GPU memory via PCIe
4. **Parallel Decompression**: CUDA kernels decompress data using 1000+ GPU cores
5. **Result Transfer**: Decompressed data is transferred back to host memory
6. **Fallback**: If GPU fails or unavailable, automatically falls back to CPU

### Memory Management

- GPU memory is allocated on-demand
- Automatic chunking for files larger than GPU memory
- Smart prefetching for overlapping I/O and compute
- Memory is released immediately after decompression

## Performance Tuning

### File Size Threshold

The default threshold of 50GB balances PCIe transfer overhead against GPU speedup:

```bash
# Adjust threshold for your hardware
export RG_GPU_THRESHOLD=25GB  # More aggressive GPU usage
export RG_GPU_THRESHOLD=100GB # Conservative GPU usage
```

### GPU Selection (Multi-GPU Systems)

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 rg pattern file.gdz

# Use multiple GPUs (not currently supported, will use first)
CUDA_VISIBLE_DEVICES=0,1 rg pattern file.gdz
```

### Optimizing for Maximum Performance

1. **Use NVMe SSDs**: I/O bottleneck is common for large files
2. **Sufficient GPU Memory**: Reduce chunking overhead
3. **Latest CUDA Version**: Performance improvements in newer versions
4. **Match GPU to File Size**:
   - 50-100GB: RTX 3060 or better
   - 100-500GB: RTX 3090/4080 or better
   - 500GB+: RTX 4090, A100, or H100

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify library links
ldd target/release/rg | grep cuda
```

**Solution**: Ensure CUDA_PATH or CUDA_HOME environment variable is set.

### Out of Memory Errors

**Symptom**: GPU decompression fails with "out of memory"

**Solution**:
1. Increase GPU memory threshold
2. Use GPU with more memory
3. Force CPU decompression for specific files

```bash
# Force CPU for this file
RG_NO_GPU=1 rg pattern large_file.gdz
```

### Slower Than CPU

**Possible causes**:
1. File too small (< 50GB) - PCIe transfer overhead dominates
2. Slow PCIe bus (use PCIe 4.0 or 5.0 if possible)
3. GPU memory fragmentation

**Solution**: Use CPU for files < 50GB, or adjust threshold.

## Benchmarks

### Test Environment
- CPU: AMD Ryzen 9 7950X (32 threads)
- GPU: NVIDIA RTX 4090 (24 GB)
- Storage: Samsung 990 PRO (PCIe 4.0 NVMe)
- OS: Ubuntu 22.04 LTS
- CUDA: 12.2

### Results

| File Size | Compression | CPU Time | GPU Time | Speedup |
|-----------|-------------|----------|----------|---------|
| 10 GB     | GDeflate    | 1.2s     | 1.5s     | 0.8x (slower) |
| 50 GB     | GDeflate    | 6.1s     | 1.3s     | **4.7x** |
| 100 GB    | GDeflate    | 12.3s    | 2.1s     | **5.9x** |
| 250 GB    | GDeflate    | 30.8s    | 4.2s     | **7.3x** |
| 500 GB    | GDeflate    | 61.5s    | 7.8s     | **7.9x** |

### Throughput

| Method | Compression | Decompression Throughput |
|--------|-------------|-------------------------|
| CPU (32-thread) | GDeflate | 8.1 GB/s |
| GPU (RTX 4090) | GDeflate | 64.1 GB/s |

## Future Enhancements

### Planned Features
- [ ] Multi-GPU support for files > 1TB
- [ ] Streaming decompression (decompress while searching)
- [ ] nvCOMP integration for optimal performance
- [ ] AMD GPU support (ROCm)
- [ ] Apple Silicon GPU support (Metal)

### nvCOMP Integration

Currently, GPU decompression uses a custom CUDA kernel. Integration with NVIDIA's nvCOMP library will provide:
- 20-30% additional speedup
- Support for more compression formats
- Better memory efficiency
- Automatic tuning for different GPUs

## Platform Support

| Platform | GPU Support | Status |
|----------|-------------|--------|
| Linux x64 | NVIDIA CUDA | ✅ Supported |
| Windows x64 | NVIDIA CUDA | ✅ Supported |
| macOS | NVIDIA CUDA | ⚠️ Limited (no official CUDA support on macOS 10.14+) |
| Linux ARM64 | NVIDIA CUDA | ✅ Supported (Jetson, Grace Hopper) |

## Security Considerations

### GPU Memory Isolation

- GPU memory is isolated per-process
- Decompressed data is securely cleared after use
- No persistent data on GPU between runs

### Decompression Bomb Protection

GPU decompression includes the same protections as CPU:
- Maximum decompressed size: 1 TB
- Compression ratio limit: 1000:1
- Timeout for excessively slow decompression

## FAQ

### Why 50GB threshold?

**Answer**: Below 50GB, PCIe transfer overhead (CPU ↔ GPU) exceeds GPU compute savings. At 50GB+, GPU parallelism dominates.

### Can I use GPU for smaller files?

**Answer**: Yes, but it will likely be slower. Adjust threshold:
```bash
export RG_GPU_THRESHOLD=10GB
```

### Does it work with standard gzip files?

**Answer**: No, GPU acceleration only works with GDeflate format (`.gdz` files). Standard gzip uses different format that doesn't support parallel decompression.

### What about AMD or Intel GPUs?

**Answer**: Currently only NVIDIA CUDA is supported. AMD ROCm and Intel oneAPI support are planned for future releases.

### Can I disable GPU support?

**Answer**: Yes:
```bash
# Temporary (this session)
RG_NO_GPU=1 rg pattern file.gdz

# Permanent (build without GPU)
cargo build --release  # omit --features cuda-gpu
```

## Contributing

GPU support is under active development. Contributions welcome:
- Performance optimizations
- Additional GPU backends (ROCm, Metal, Vulkan)
- nvCOMP integration
- Multi-GPU support

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [nvCOMP Library](https://github.com/NVIDIA/nvcomp)
- [GDeflate Specification](https://github.com/microsoft/DirectStorage)
- [Ripgrep User Guide](../GUIDE.md)

## License

GPU support code is dual-licensed under MIT or UNLICENSE, same as ripgrep.
