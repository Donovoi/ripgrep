//! Example demonstrating NVIDIA GPU-accelerated decompression
//!
//! This example shows how to use GPU acceleration for large file decompression.
//!
//! To run this example with GPU support:
//! ```
//! cargo run --release --features cuda-gpu --example gpu_demo
//! ```
//!
//! Without GPU support:
//! ```
//! cargo run --release --example gpu_demo
//! ```

use std::time::Instant;

fn main() {
    println!("=== GDeflate GPU Acceleration Demo ===\n");

    // Check if GPU support is compiled in
    #[cfg(feature = "cuda-gpu")]
    {
        println!("✓ GPU support is ENABLED");
        demo_with_gpu();
    }

    #[cfg(not(feature = "cuda-gpu"))]
    {
        println!("✗ GPU support is DISABLED");
        println!("  To enable GPU support, rebuild with:");
        println!("  cargo build --release --features cuda-gpu\n");
        demo_without_gpu();
    }
}

#[cfg(feature = "cuda-gpu")]
fn demo_with_gpu() {
    use gdeflate::gpu::{
        get_gpu_devices, is_gpu_available, should_use_gpu, GPU_SIZE_THRESHOLD,
    };

    println!("\n--- GPU Device Information ---");

    if is_gpu_available() {
        println!("✓ NVIDIA GPU detected and available");

        let devices = get_gpu_devices();
        for (i, device) in devices.iter().enumerate() {
            println!("\nGPU {}: {}", i, device.name);
            println!(
                "  Total Memory: {:.2} GB",
                device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!(
                "  Free Memory: {:.2} GB",
                device.free_memory as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!(
                "  Compute Capability: {}.{}",
                device.compute_capability.0, device.compute_capability.1
            );
            println!("  Multiprocessors: {}", device.multiprocessor_count);
        }
    } else {
        println!("✗ No NVIDIA GPU detected");
        println!("  Possible reasons:");
        println!("  - CUDA Toolkit not installed");
        println!("  - NVIDIA drivers not installed");
        println!("  - No compatible NVIDIA GPU in system");
        println!("\n  Falling back to CPU decompression");
    }

    println!("\n--- GPU Usage Heuristics ---");
    println!(
        "GPU Threshold: {} GB",
        GPU_SIZE_THRESHOLD / (1024 * 1024 * 1024)
    );

    let test_sizes = vec![
        ("1 GB", 1 * 1024 * 1024 * 1024),
        ("10 GB", 10 * 1024 * 1024 * 1024),
        ("50 GB", 50 * 1024 * 1024 * 1024),
        ("100 GB", 100 * 1024 * 1024 * 1024),
        ("500 GB", 500 * 1024 * 1024 * 1024),
    ];

    for (label, size) in test_sizes {
        let use_gpu = should_use_gpu(size);
        let method = if use_gpu { "GPU" } else { "CPU" };
        println!("  {} file: {} decompression", label, method);
    }

    demo_compression_decompression_with_gpu();
}

#[cfg(feature = "cuda-gpu")]
fn demo_compression_decompression_with_gpu() {
    use gdeflate::{compress, decompress, decompress_auto};

    println!("\n--- Compression/Decompression Demo ---");

    // Create test data
    let test_data = create_test_data(10 * 1024 * 1024); // 10 MB
    println!("Test data size: {} MB", test_data.len() / (1024 * 1024));

    // Compress
    let start = Instant::now();
    let compressed = compress(&test_data, 6, 0).expect("Compression failed");
    let compress_time = start.elapsed();

    let ratio = (compressed.len() as f64 / test_data.len() as f64) * 100.0;
    println!(
        "Compressed to {} bytes ({:.2}% ratio) in {:.2?}",
        compressed.len(),
        ratio,
        compress_time
    );

    // Decompress with CPU
    let start = Instant::now();
    let decompressed_cpu = decompress(&compressed, test_data.len(), 0)
        .expect("CPU decompression failed");
    let cpu_time = start.elapsed();
    println!("CPU decompression: {:.2?}", cpu_time);

    // Decompress with auto (will use GPU if file >= 50GB)
    let start = Instant::now();
    let decompressed_auto = decompress_auto(&compressed, test_data.len())
        .expect("Auto decompression failed");
    let auto_time = start.elapsed();
    println!(
        "Auto decompression: {:.2?} (used CPU for small file)",
        auto_time
    );

    // Verify results
    assert_eq!(test_data, decompressed_cpu, "CPU decompression mismatch!");
    assert_eq!(test_data, decompressed_auto, "Auto decompression mismatch!");
    println!("✓ All decompression results verified correctly");

    println!("\n--- Performance Notes ---");
    println!("• This 10 MB file uses CPU decompression (< 50 GB threshold)");
    println!("• For files >= 50 GB, GPU would provide 4-15x speedup");
    println!(
        "• GPU decompression automatically falls back to CPU if unavailable"
    );
}

#[cfg(not(feature = "cuda-gpu"))]
fn demo_without_gpu() {
    use gdeflate::{compress, decompress};

    println!("\n--- CPU-Only Compression/Decompression Demo ---");

    // Create test data
    let test_data = create_test_data(10 * 1024 * 1024); // 10 MB
    println!("Test data size: {} MB", test_data.len() / (1024 * 1024));

    // Compress
    let start = Instant::now();
    let compressed = compress(&test_data, 6, 0).expect("Compression failed");
    let compress_time = start.elapsed();

    let ratio = (compressed.len() as f64 / test_data.len() as f64) * 100.0;
    println!(
        "Compressed to {} bytes ({:.2}% ratio) in {:.2?}",
        compressed.len(),
        ratio,
        compress_time
    );

    // Decompress with CPU
    let start = Instant::now();
    let decompressed = decompress(&compressed, test_data.len(), 0)
        .expect("CPU decompression failed");
    let cpu_time = start.elapsed();
    println!("CPU decompression: {:.2?}", cpu_time);

    // Verify
    assert_eq!(test_data, decompressed, "Decompression mismatch!");
    println!("✓ Decompression verified correctly");

    println!("\n--- Performance Notes ---");
    println!(
        "• CPU-only build provides excellent performance for most use cases"
    );
    println!(
        "• For extremely large files (>= 50 GB), consider GPU acceleration"
    );
    println!("• GPU provides 4-15x speedup for 50GB-500GB files");
}

// Helper function to create test data
fn create_test_data(size: usize) -> Vec<u8> {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);

    while data.len() < size {
        data.extend_from_slice(pattern);
    }

    data.truncate(size);
    data
}
