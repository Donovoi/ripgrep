#!/bin/bash
# Benchmark script for comparing GDeflate vs traditional compression with ripgrep
# This script helps evaluate the performance benefits of DirectStorage/GDeflate integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BENCHMARK_DIR="${BENCHMARK_DIR:-./benchmark_data}"
RESULTS_FILE="${RESULTS_FILE:-./benchmark_results.csv}"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  DirectStorage/GDeflate Benchmark Suite for Ripgrep      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    local missing_tools=()
    
    if ! command -v rg &> /dev/null; then
        missing_tools+=("ripgrep (rg)")
    fi
    
    if ! command -v gzip &> /dev/null; then
        missing_tools+=("gzip")
    fi
    
    if ! command -v time &> /dev/null; then
        missing_tools+=("time")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo -e "${RED}Error: Missing required tools:${NC}"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        exit 1
    fi
    
    echo -e "${GREEN}✓ All prerequisites met${NC}\n"
}

# Create test data
create_test_data() {
    echo -e "${YELLOW}Creating test data...${NC}"
    
    mkdir -p "$BENCHMARK_DIR"
    
    # Create different sizes of test files
    # Small files (10KB - 100KB)
    for i in {1..10}; do
        local size=$((10 + RANDOM % 90))
        head -c "${size}K" /dev/urandom | base64 > "$BENCHMARK_DIR/small_${i}.txt"
    done
    
    # Medium files (1MB - 10MB)
    for i in {1..5}; do
        local size=$((1 + RANDOM % 9))
        head -c "${size}M" /dev/urandom | base64 > "$BENCHMARK_DIR/medium_${i}.txt"
    done
    
    # Large files (10MB - 50MB)
    for i in {1..3}; do
        local size=$((10 + RANDOM % 40))
        head -c "${size}M" /dev/urandom | base64 > "$BENCHMARK_DIR/large_${i}.txt"
    done
    
    # Add some pattern-rich files (easier to compress, more realistic)
    cat > "$BENCHMARK_DIR/source_code.txt" << 'EOF'
// Example source code for testing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    printf("Hello, world!\n");
    for (int i = 0; i < 1000; i++) {
        printf("Iteration %d\n", i);
    }
    return 0;
}
EOF
    
    # Replicate the source code file to make it larger
    for i in {1..100}; do
        cat "$BENCHMARK_DIR/source_code.txt" >> "$BENCHMARK_DIR/code_large.txt"
    done
    
    echo -e "${GREEN}✓ Test data created in $BENCHMARK_DIR${NC}\n"
}

# Compress test files
compress_test_files() {
    echo -e "${YELLOW}Compressing test files with gzip...${NC}"
    
    for file in "$BENCHMARK_DIR"/*.txt; do
        if [[ ! -f "${file}.gz" ]]; then
            gzip -k -9 "$file"
            echo "  Compressed: $(basename "$file")"
        fi
    done
    
    echo -e "${GREEN}✓ Compression complete${NC}\n"
    
    # Note: GDeflate compression would be done here if the tool is available
    echo -e "${YELLOW}Note: GDeflate compression requires the GDeflateDemo tool${NC}"
    echo -e "${YELLOW}      from https://github.com/Donovoi/DirectStorage${NC}\n"
}

# Run benchmark for a single file
benchmark_file() {
    local file=$1
    local pattern=$2
    local format=$3  # "uncompressed", "gzip", or "gdeflate"
    
    local test_file=""
    case "$format" in
        "uncompressed")
            test_file="$file"
            ;;
        "gzip")
            test_file="${file}.gz"
            ;;
        "gdeflate")
            test_file="${file}.gdz"
            ;;
    esac
    
    if [[ ! -f "$test_file" ]]; then
        echo "0,0,0,skipped"
        return
    fi
    
    local size
    size=$(stat -f%z "$test_file" 2>/dev/null || stat -c%s "$test_file" 2>/dev/null)
    
    # Run benchmark 3 times and take average
    local total_time=0
    local runs=3
    
    for _ in $(seq 1 $runs); do
        local start
        start=$(date +%s%N)
        rg "$pattern" "$test_file" > /dev/null 2>&1 || true
        local end
        end=$(date +%s%N)
        local elapsed=$(( (end - start) / 1000000 )) # Convert to milliseconds
        total_time=$((total_time + elapsed))
    done
    
    local avg_time=$((total_time / runs))
    
    echo "$size,$avg_time,$runs,success"
}

# Run comprehensive benchmarks
run_benchmarks() {
    echo -e "${YELLOW}Running benchmarks...${NC}"
    echo -e "${BLUE}This may take several minutes...${NC}\n"
    
    # Create results file with header
    echo "File,Format,Size_Bytes,Avg_Time_MS,Runs,Status" > "$RESULTS_FILE"
    
    local pattern="the"  # Common pattern to search for
    
    # Benchmark each file in different formats
    for file in "$BENCHMARK_DIR"/*.txt; do
        local basename
        basename=$(basename "$file")
        echo -e "  Testing ${basename}..."
        
        # Uncompressed
        local result
        result=$(benchmark_file "$file" "$pattern" "uncompressed")
        echo "${basename},uncompressed,$result" >> "$RESULTS_FILE"
        
        # Gzip
        if [[ -f "${file}.gz" ]]; then
            result=$(benchmark_file "$file" "$pattern" "gzip")
            echo "${basename},gzip,$result" >> "$RESULTS_FILE"
        fi
        
        # GDeflate (if available)
        if [[ -f "${file}.gdz" ]]; then
            result=$(benchmark_file "$file" "$pattern" "gdeflate")
            echo "${basename},gdeflate,$result" >> "$RESULTS_FILE"
        fi
    done
    
    echo -e "\n${GREEN}✓ Benchmarks complete${NC}"
    echo -e "Results saved to: ${BLUE}$RESULTS_FILE${NC}\n"
}

# Analyze results
analyze_results() {
    echo -e "${YELLOW}Analyzing results...${NC}\n"
    
    if [[ ! -f "$RESULTS_FILE" ]]; then
        echo -e "${RED}Error: Results file not found${NC}"
        return
    fi
    
    echo -e "${BLUE}Performance Summary:${NC}"
    echo -e "${BLUE}════════════════════${NC}\n"
    
    # Calculate average times by format
    awk -F',' 'NR>1 && $6=="success" {
        sum[$2] += $4; 
        count[$2]++; 
        size[$2] += $3
    } 
    END {
        for (format in sum) {
            avg_time = sum[format] / count[format];
            avg_size = size[format] / count[format];
            throughput = (avg_size / 1024 / 1024) / (avg_time / 1000);
            printf "%-15s: %8.2f ms average, %8.2f MB/s throughput\n", 
                   format, avg_time, throughput
        }
    }' "$RESULTS_FILE"
    
    echo ""
    
    # Calculate speedup factors
    echo -e "${BLUE}Speedup Factors (compared to uncompressed):${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}\n"
    
    awk -F',' 'NR>1 && $6=="success" {
        file=$1; 
        format=$2; 
        time=$4;
        times[file,format] = time
    } 
    END {
        for (file_format in times) {
            split(file_format, parts, SUBSEP);
            file = parts[1];
            format = parts[2];
            if (format == "uncompressed") {
                baseline[file] = times[file_format];
            }
        }
        
        for (file_format in times) {
            split(file_format, parts, SUBSEP);
            file = parts[1];
            format = parts[2];
            if (format != "uncompressed" && baseline[file] > 0) {
                speedup = times[file_format] / baseline[file];
                improvement = (1 - speedup) * 100;
                printf "%-30s %-10s: %5.2fx slower (-%5.1f%%)\n", 
                       file, format, speedup, improvement
            }
        }
    }' "$RESULTS_FILE" | sort
    
    echo ""
}

# Generate report
generate_report() {
    local report_file="benchmark_report.md"
    
    echo -e "${YELLOW}Generating detailed report...${NC}"
    
    cat > "$report_file" << 'EOF'
# DirectStorage/GDeflate Integration Benchmark Report

## Test Configuration

- **Date**: $(date)
- **System**: $(uname -a)
- **Ripgrep Version**: $(rg --version | head -1)

## Methodology

This benchmark compares search performance across different compression formats:
- **Uncompressed**: Direct file reading (baseline)
- **Gzip**: Traditional external process decompression (current ripgrep)
- **GDeflate**: Proposed native parallel decompression (future ripgrep)

### Test Scenarios

1. **Small files** (10KB - 100KB): Tests process overhead
2. **Medium files** (1MB - 10MB): Tests decompression throughput
3. **Large files** (10MB - 50MB): Tests parallel decompression scaling

Each test was run 3 times and averaged to reduce noise.

## Results

EOF
    
    # Append CSV results
    {
        echo "### Raw Data"
        echo ""
        echo '```'
        cat "$RESULTS_FILE"
        echo '```'
        echo ""
    } >> "$report_file"
    
    # Append analysis
    echo "### Analysis" >> "$report_file"
    echo "" >> "$report_file"
    analyze_results >> "$report_file" 2>&1
    
    echo -e "${GREEN}✓ Report generated: ${BLUE}$report_file${NC}\n"
}

# Main execution
main() {
    check_prerequisites
    
    # Check if data already exists
    if [[ ! -d "$BENCHMARK_DIR" ]] || [[ -z "$(ls -A "$BENCHMARK_DIR")" ]]; then
        create_test_data
        compress_test_files
    else
        echo -e "${YELLOW}Using existing test data in $BENCHMARK_DIR${NC}\n"
    fi
    
    run_benchmarks
    analyze_results
    generate_report
    
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Benchmark Complete!                                      ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Results: ${BLUE}$RESULTS_FILE${NC}"
    echo -e "Report:  ${BLUE}benchmark_report.md${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Review the benchmark results"
    echo "2. If GDeflate integration shows promise, proceed with implementation"
    echo "3. Test on real-world codebases and data"
    echo ""
}

# Run main function
main "$@"
