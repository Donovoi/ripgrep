use gdeflate::{compress, compress_bound, decompress, version};

fn main() {
    println!("GDeflate version: {}", version());
    println!();

    // Example text to compress
    let input = b"Hello, world! This is a test of GDeflate compression. \
                  The quick brown fox jumps over the lazy dog. \
                  GDeflate is a GPU-optimized compression format that closely matches DEFLATE.";

    println!("Original size: {} bytes", input.len());
    println!("Max compressed size: {} bytes", compress_bound(input.len()));
    println!();

    // Test different compression levels
    for level in [1, 6, 12] {
        let compressed = compress(input, level, 0)
            .expect(&format!("compression failed at level {}", level));

        let ratio = (compressed.len() as f64 / input.len() as f64) * 100.0;
        println!(
            "Level {}: {} bytes ({:.1}% of original)",
            level,
            compressed.len(),
            ratio
        );

        // Verify decompression
        let decompressed = decompress(&compressed, input.len(), 0)
            .expect(&format!("decompression failed at level {}", level));

        assert_eq!(
            input,
            decompressed.as_slice(),
            "Decompressed data doesn't match original at level {}",
            level
        );
    }

    println!();
    println!("All compression levels tested successfully!");
}
