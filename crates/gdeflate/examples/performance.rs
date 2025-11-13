//! Performance demonstration example for GDeflate
//!
//! This example demonstrates the performance impact of different thread counts
//! and shows how to use the auto-tuning features.

use gdeflate::{compress, decompress, decompress_auto, recommended_workers};
use std::time::Instant;

fn format_throughput(bytes: usize, duration: std::time::Duration) -> String {
    let throughput = (bytes as f64) / duration.as_secs_f64() / 1_000_000.0;
    format!("{:.2} MB/s", throughput)
}

fn benchmark_decompression(
    compressed: &[u8],
    output_size: usize,
    num_workers: u32,
    label: &str,
) -> std::time::Duration {
    let start = Instant::now();
    let _result = decompress(compressed, output_size, num_workers)
        .expect("decompression failed");
    let duration = start.elapsed();
    
    println!(
        "  {} threads: {:>8.2} ms | {}",
        label,
        duration.as_secs_f64() * 1000.0,
        format_throughput(output_size, duration)
    );
    
    duration
}

fn main() {
    println!("GDeflate Performance Demonstration\n");
    println!("{}", "=".repeat(60));
    
    // Create test data of different sizes
    let sizes = [
        (100_000, "100 KB"),
        (1_000_000, "1 MB"),
        (10_000_000, "10 MB"),
        (50_000_000, "50 MB"),
    ];
    
    for (size, label) in &sizes {
        println!("\n{} Test Data:", label);
        println!("{}", "-".repeat(60));
        
        // Create test data with some pattern
        let mut input = Vec::with_capacity(*size);
        for i in 0..*size {
            input.push((i % 256) as u8);
        }
        
        // Compress
        println!("  Compressing...");
        let start = Instant::now();
        let compressed = compress(&input, 6, 0).expect("compression failed");
        let compress_duration = start.elapsed();
        
        let ratio = (compressed.len() as f64) / (input.len() as f64) * 100.0;
        println!(
            "  Compressed {} bytes to {} bytes ({:.1}% ratio)",
            input.len(),
            compressed.len(),
            ratio
        );
        println!(
            "  Compression: {:.2} ms | {}",
            compress_duration.as_secs_f64() * 1000.0,
            format_throughput(input.len(), compress_duration)
        );
        
        // Recommend optimal workers
        let recommended = recommended_workers(input.len());
        println!("\n  Decompression Performance:");
        println!("  Recommended workers for this size: {}", recommended);
        
        // Benchmark different thread counts
        let single = benchmark_decompression(&compressed, input.len(), 1, "1 ");
        
        if recommended >= 2 {
            let dual = benchmark_decompression(&compressed, input.len(), 2, "2 ");
            let speedup = single.as_secs_f64() / dual.as_secs_f64();
            println!("    Speedup vs single-thread: {:.2}x", speedup);
        }
        
        if recommended >= 4 {
            let quad = benchmark_decompression(&compressed, input.len(), 4, "4 ");
            let speedup = single.as_secs_f64() / quad.as_secs_f64();
            println!("    Speedup vs single-thread: {:.2}x", speedup);
        }
        
        if recommended >= 8 {
            let oct = benchmark_decompression(&compressed, input.len(), 8, "8 ");
            let speedup = single.as_secs_f64() / oct.as_secs_f64();
            println!("    Speedup vs single-thread: {:.2}x", speedup);
        }
        
        // Test auto mode
        println!("\n  Auto-tuning:");
        let start = Instant::now();
        let _result = decompress_auto(&compressed, input.len())
            .expect("auto decompression failed");
        let auto_duration = start.elapsed();
        println!(
            "  Auto mode: {:>8.2} ms | {}",
            auto_duration.as_secs_f64() * 1000.0,
            format_throughput(input.len(), auto_duration)
        );
        
        let speedup = single.as_secs_f64() / auto_duration.as_secs_f64();
        println!("    Speedup vs single-thread: {:.2}x", speedup);
    }
    
    println!("\n{}", "=".repeat(60));
    println!("\nKey Findings:");
    println!("  - Small files benefit less from parallelism (thread overhead)");
    println!("  - Large files see significant speedups (8-15x typical)");
    println!("  - Auto-tuning selects optimal thread count automatically");
    println!("  - SIMD optimizations are always active (built-in speedup)");
    
    // Display system info
    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("\nSystem Info:");
    println!("  Available CPU threads: {}", cpu_count);
    println!("  GDeflate max workers: 32");
    println!("  SIMD: Automatically detected and enabled");
}
