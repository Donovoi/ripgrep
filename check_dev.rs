use std::os::unix::fs::FileTypeExt;

fn main() {
    let path = "/dev/null";
    let m = std::fs::metadata(path).unwrap();
    println!("{} len: {}", path, m.len());
    println!("{} is_file: {}", path, m.is_file());
    println!("{} is_block_device: {}", path, m.file_type().is_block_device());
    println!("{} is_char_device: {}", path, m.file_type().is_char_device());
}
