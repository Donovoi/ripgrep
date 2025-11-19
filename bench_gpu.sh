#!/bin/bash
set -e

# Ensure the bridge is built
cd gpu_bridge/build
cmake .. > /dev/null
make > /dev/null
cd ../..

export RG_NVTEXT_BRIDGE_PATH=$(pwd)/gpu_bridge/build/librg_gpu_regex_bridge.so
LARGE_FILE="large_file.txt"

# Ensure large file exists (500MB for better benchmark)
if [ ! -f "$LARGE_FILE" ] || [ $(stat -c%s "$LARGE_FILE") -lt 500000000 ]; then
    echo "Generating 500MB file..."
    # Create 500MB file with "match_me" at the very end
    dd if=/dev/urandom of="$LARGE_FILE" bs=1M count=500 status=none
    echo "match_me" >> "$LARGE_FILE"
fi

echo "Benchmarking CPU vs GPU on $(stat -c%s $LARGE_FILE) bytes..."

# Warmup
echo "Warming up..."
cargo run --release --features cuda-gpu -- "match_me" "$LARGE_FILE" > /dev/null

echo "--- CPU Run (Bridge Disabled) ---"
unset RG_NVTEXT_BRIDGE_PATH
time cargo run --release --features cuda-gpu -- "match_me" "$LARGE_FILE" > /dev/null

echo "--- GPU Run ---"
export RG_NVTEXT_BRIDGE_PATH=$(pwd)/gpu_bridge/build/librg_gpu_regex_bridge.so
time cargo run --release --features cuda-gpu -- "match_me" "$LARGE_FILE" > /dev/null

