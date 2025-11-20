#!/bin/bash
# GPU Performance Benchmark Suite
# Tests GPU acceleration performance vs CPU for various file sizes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/testdata"
RESULTS_FILE="${SCRIPT_DIR}/gpu_benchmark_results.md"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================"
echo "    GPU Performance Benchmark Suite"
echo "============================================"
echo ""

# Check if ripgrep is built with GPU support
if ! ./target/release/rg --version 2>&1 | grep -q "cuda-gpu"; then
    echo "ERROR: ripgrep not built with cuda-gpu feature"
    echo "Build with: cargo build --release --features cuda-gpu"
    exit 1
fi

# Create test directory
mkdir -p "$TEST_DIR"

# Function to format bytes to human-readable
format_bytes() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt $((1024*1024)) ]; then
        echo "$((bytes/1024))KB"
    elif [ $bytes -lt $((1024*1024*1024)) ]; then
        echo "$((bytes/(1024*1024)))MB"
    else
        echo "$((bytes/(1024*1024*1024)))GB"
    fi
}

# Function to run benchmark
benchmark_file() {
    local size=$1
    local size_name=$2
    local pattern=$3
    
    echo -e "${BLUE}Benchmarking ${size_name} file...${NC}"
    
    local test_file="${TEST_DIR}/test_${size_name}.txt"
    local compressed_file="${test_file}.gz"
    
    # Generate test file if it doesn't exist
    if [ ! -f "$test_file" ]; then
        echo "Generating ${size_name} test file..."
        # Generate file with repeated pattern for realistic compression
        local lines=$((size / 100))
        for i in $(seq 1 $lines); do
            echo "This is line $i with some ERROR and WARNING messages to search for pattern matches" >> "$test_file"
        done
    fi
    
    # Compress the file
    if [ ! -f "$compressed_file" ]; then
        echo "Compressing ${size_name} file..."
        gzip -c "$test_file" > "$compressed_file"
    fi
    
    local actual_size=$(stat -f%z "$test_file" 2>/dev/null || stat -c%s "$test_file")
    echo "Actual file size: $(format_bytes $actual_size)"
    
    # Benchmark CPU (without GPU flag)
    echo "Running CPU benchmark..."
    local cpu_start=$(date +%s%N)
    RG_NO_GPU=1 ./target/release/rg -z "$pattern" "$compressed_file" > /dev/null 2>&1 || true
    local cpu_end=$(date +%s%N)
    local cpu_time=$(((cpu_end - cpu_start) / 1000000)) # Convert to milliseconds
    
    # Benchmark GPU (with GPU flag)
    echo "Running GPU benchmark..."
    local gpu_start=$(date +%s%N)
    ./target/release/rg -z --gpu "$pattern" "$compressed_file" > /dev/null 2>&1 || true
    local gpu_end=$(date +%s%N)
    local gpu_time=$(((gpu_end - gpu_start) / 1000000)) # Convert to milliseconds
    
    # Calculate speedup
    local speedup="N/A"
    if [ $gpu_time -gt 0 ]; then
        speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
    fi
    
    echo -e "${GREEN}CPU time: ${cpu_time}ms${NC}"
    echo -e "${GREEN}GPU time: ${gpu_time}ms${NC}"
    echo -e "${GREEN}Speedup: ${speedup}x${NC}"
    echo ""
    
    # Append to results file
    echo "| ${size_name} | $(format_bytes $actual_size) | ${cpu_time}ms | ${gpu_time}ms | ${speedup}x |" >> "$RESULTS_FILE"
}

# System information
echo "System Configuration:"
echo "===================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1
else
    echo "No NVIDIA GPU detected (nvidia-smi not available)"
fi
lscpu | grep "Model name" || sysctl -n machdep.cpu.brand_string || echo "CPU info not available"
echo ""

# Initialize results file
cat > "$RESULTS_FILE" << EOF
# GPU Benchmark Results

**Date**: $(date)
**System**: $(uname -a)

## System Configuration

EOF

if command -v nvidia-smi &> /dev/null; then
    echo "**GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "$RESULTS_FILE"
    echo "**GPU Memory**: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)" >> "$RESULTS_FILE"
fi

echo "**CPU**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs || sysctl -n machdep.cpu.brand_string || echo "Unknown")" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "## Benchmark Results" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "| File Size | Actual Size | CPU Time | GPU Time | Speedup |" >> "$RESULTS_FILE"
echo "|-----------|-------------|----------|----------|---------|" >> "$RESULTS_FILE"

# Run benchmarks for different file sizes
PATTERN="ERROR"

# Small file (should show CPU is faster due to overhead)
benchmark_file $((1 * 1024 * 1024)) "1MB" "$PATTERN"

# Medium file (GPU may start to show benefit)
benchmark_file $((100 * 1024 * 1024)) "100MB" "$PATTERN"

# Large file (GPU should show clear benefit if available)
benchmark_file $((1024 * 1024 * 1024)) "1GB" "$PATTERN"

# Add analysis section
cat >> "$RESULTS_FILE" << 'EOF'

## Analysis

### Expected Results

- **Small files (<10MB)**: CPU should be faster due to PCIe transfer overhead
- **Medium files (10MB-10GB)**: GPU may show 10-30% improvement
- **Large files (>10GB)**: GPU should show 2-5x improvement
- **Huge files (>50GB)**: GPU should show 4-10x improvement

### Notes

- GPU acceleration is designed for very large files (50GB+)
- For typical ripgrep usage (<1MB files), CPU is faster
- PCIe transfer time limits GPU benefit for smaller files
- Results depend heavily on:
  - GPU model (compute capability, memory bandwidth)
  - Storage speed (NVMe vs SATA)
  - Pattern complexity
  - File compressibility

### Recommendations

- Use GPU acceleration for files >10GB when available
- For typical code search (<1MB files), use default CPU path
- Consider storage I/O as potential bottleneck
- GPU memory size limits maximum file size

EOF

echo "============================================"
echo "Benchmark complete!"
echo "Results saved to: $RESULTS_FILE"
echo "============================================"
echo ""
echo "Summary:"
cat "$RESULTS_FILE" | grep "^|" | head -5
