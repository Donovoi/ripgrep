#[cfg(feature = "gdeflate")]
#[test]
fn test_gdeflate_search() {
    use std::fs::File;
    use std::io::Write;

    let (dir, mut cmd) = crate::util::setup("test_gdeflate_search");

    // Create test content
    let content = b"Test line with Sherlock Holmes.\n\
                     Another line with Watson.\n\
                     And a line with Moriarty.\n";

    // Compress with gdeflate
    let compressed =
        gdeflate::compress(content, 6, 0).expect("compression failed");

    // Create GDZ file with magic number and size header
    let test_file = dir.path().join("test.gdz");
    let mut file = File::create(&test_file).expect("failed to create file");

    // Write magic number: "GDZ\0"
    file.write_all(b"GDZ\0").expect("failed to write magic");

    // Write uncompressed size (little-endian u64)
    file.write_all(&(content.len() as u64).to_le_bytes())
        .expect("failed to write size");

    // Write compressed data
    file.write_all(&compressed).expect("failed to write compressed data");
    drop(file);

    // Test searching with ripgrep
    let output = cmd.arg("-z").arg("Sherlock").arg("test.gdz").stdout();

    assert!(
        output.contains("Sherlock Holmes"),
        "output should contain 'Sherlock Holmes'"
    );
}

#[cfg(feature = "gdeflate")]
#[test]
fn test_gdeflate_with_gz_extension() {
    use std::fs::File;
    use std::io::Write;

    let (dir, mut cmd) = crate::util::setup("test_gdeflate_gz");

    // Create test content
    let content = b"GDeflate test with .gz extension\n";

    // Compress with gdeflate
    let compressed =
        gdeflate::compress(content, 6, 0).expect("compression failed");

    // Create file with .gz extension but GDeflate format
    let test_file = dir.path().join("test.gz");
    let mut file = File::create(&test_file).expect("failed to create file");

    // Write GDeflate format
    file.write_all(b"GDZ\0").expect("failed to write magic");
    file.write_all(&(content.len() as u64).to_le_bytes())
        .expect("failed to write size");
    file.write_all(&compressed).expect("failed to write compressed data");
    drop(file);

    // Test searching - should use GDeflate decompression based on magic, not extension
    let output = cmd.arg("-z").arg("GDeflate").arg("test.gz").stdout();

    assert!(
        output.contains("GDeflate test"),
        "output should contain 'GDeflate test'"
    );
}
