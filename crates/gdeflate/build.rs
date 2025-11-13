use std::env;
use std::path::PathBuf;

/// Check if nvCOMP library is available
fn check_nvcomp_available(cuda_path: &str) -> bool {
    // Check environment variable first
    if let Ok(nvcomp_root) = env::var("NVCOMP_ROOT") {
        let nvcomp_include = PathBuf::from(&nvcomp_root).join("include");
        let nvcomp_header = nvcomp_include.join("nvcomp.h");
        if nvcomp_header.exists() {
            println!("cargo:warning=Found nvCOMP at NVCOMP_ROOT: {}", nvcomp_root);
            println!("cargo:rustc-link-search=native={}/lib", nvcomp_root);
            println!("cargo:rustc-link-search=native={}/lib64", nvcomp_root);
            return true;
        }
    }
    
    // Check CUDA installation
    let cuda_nvcomp_include = PathBuf::from(cuda_path).join("include").join("nvcomp.h");
    if cuda_nvcomp_include.exists() {
        println!("cargo:warning=Found nvCOMP in CUDA installation");
        return true;
    }
    
    // Check common system paths
    let system_paths = vec![
        "/usr/local/nvcomp/include/nvcomp.h",
        "/usr/include/nvcomp.h",
        "/opt/nvcomp/include/nvcomp.h",
    ];
    
    for path_str in system_paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            println!("cargo:warning=Found nvCOMP at system path: {}", path_str);
            if let Some(parent) = path.parent() {
                if let Some(grandparent) = parent.parent() {
                    println!("cargo:rustc-link-search=native={}/lib", grandparent.display());
                    println!("cargo:rustc-link-search=native={}/lib64", grandparent.display());
                }
            }
            return true;
        }
    }
    
    false
}

fn main() {
    let gdeflate_src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("GDeflate");
    let libdeflate_src =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("3rdparty")
            .join("libdeflate");

    // Check if libdeflate submodule is initialized
    if !libdeflate_src.join("lib").exists() {
        panic!(
            "libdeflate submodule not initialized!\n\
             Please run: git submodule update --init --recursive\n\
             from the repository root directory."
        );
    }

    // Build libdeflate
    let libdeflate_files = [
        "lib/adler32.c",
        "lib/crc32.c",
        "lib/deflate_compress.c",
        "lib/deflate_decompress.c",
        "lib/gdeflate_compress.c",
        "lib/gdeflate_decompress.c",
        "lib/gzip_compress.c",
        "lib/gzip_decompress.c",
        "lib/utils.c",
        "lib/zlib_compress.c",
        "lib/zlib_decompress.c",
        "lib/x86/cpu_features.c",
    ];

    let mut libdeflate_build = cc::Build::new();
    for file in &libdeflate_files {
        libdeflate_build.file(libdeflate_src.join(file));
    }
    libdeflate_build.include(&libdeflate_src).warnings(false);

    // Platform-specific flags
    if cfg!(target_os = "windows") {
        libdeflate_build.define("_CRT_SECURE_NO_WARNINGS", None);
    }

    libdeflate_build.compile("deflate");

    // Build GDeflate
    let gdeflate_files =
        ["GDeflateCompress.cpp", "GDeflateDecompress.cpp", "GDeflate_c.cpp"];

    let mut gdeflate_build = cc::Build::new();
    for file in &gdeflate_files {
        gdeflate_build.file(gdeflate_src.join(file));
    }
    gdeflate_build
        .cpp(true)
        .std("c++17")
        .include(&gdeflate_src)
        .include(&libdeflate_src)
        .warnings(false);

    // Platform-specific settings
    if cfg!(target_os = "windows") {
        gdeflate_build.define("_CRT_SECURE_NO_WARNINGS", None);
    }

    gdeflate_build.compile("gdeflate");

    // Build GPU support if cuda-gpu feature is enabled
    if cfg!(feature = "cuda-gpu") {
        println!("cargo:warning=Building with NVIDIA GPU support");
        
        // Check for CUDA toolkit
        let cuda_path = env::var("CUDA_PATH")
            .or_else(|_| env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        
        let cuda_include = PathBuf::from(&cuda_path).join("include");
        let cuda_lib = PathBuf::from(&cuda_path).join("lib64");
        
        // Check if CUDA is actually available
        let cuda_available = cuda_include.join("cuda_runtime.h").exists();
        
        if cuda_available {
            println!("cargo:warning=CUDA toolkit found at {}", cuda_path);
            
            // Check for nvCOMP library
            let nvcomp_available = check_nvcomp_available(&cuda_path);
            
            // Build GPU support module with CUDA
            let mut gpu_build = cc::Build::new();
            gpu_build
                .file(gdeflate_src.join("GDeflate_gpu.cpp"))
                .cpp(true)
                .std("c++17")
                .include(&gdeflate_src)
                .include(&cuda_include)
                .define("CUDA_GPU_SUPPORT", None)
                .warnings(false);
            
            if nvcomp_available {
                println!("cargo:warning=nvCOMP library found, enabling GPU decompression");
                gpu_build.define("NVCOMP_AVAILABLE", None);
                
                // Add nvCOMP source file
                gpu_build.file(gdeflate_src.join("GDeflate_nvcomp.cpp"));
                
                // Add nvCOMP include path
                let nvcomp_include = PathBuf::from(&cuda_path).join("include");
                gpu_build.include(&nvcomp_include);
            } else {
                println!("cargo:warning=nvCOMP library not found, using stub GPU module");
                println!("cargo:warning=GPU will fallback to CPU at runtime");
                println!("cargo:warning=To enable GPU decompression, install nvCOMP library");
            }
            
            // Platform-specific settings
            if cfg!(target_os = "windows") {
                gpu_build.define("_CRT_SECURE_NO_WARNINGS", None);
            }
            
            gpu_build.compile("gdeflate_gpu");
            
            // Link CUDA runtime
            println!("cargo:rustc-link-search=native={}", cuda_lib.display());
            println!("cargo:rustc-link-lib=cudart");
            
            // Link nvCOMP if available
            if nvcomp_available {
                println!("cargo:rustc-link-lib=nvcomp");
                println!("cargo:rustc-link-lib=nvcomp_gdeflate");
            }
        } else {
            println!("cargo:warning=CUDA toolkit not found, building stub GPU module");
            println!("cargo:warning=GPU acceleration will not be available at runtime");
            println!("cargo:warning=To enable GPU support, install CUDA Toolkit 11.0+ and set CUDA_PATH or CUDA_HOME");
            
            // Build stub GPU support module without CUDA
            let mut gpu_build = cc::Build::new();
            gpu_build
                .file(gdeflate_src.join("GDeflate_gpu.cpp"))
                .cpp(true)
                .std("c++17")
                .include(&gdeflate_src)
                .warnings(false);
            
            // Platform-specific settings
            if cfg!(target_os = "windows") {
                gpu_build.define("_CRT_SECURE_NO_WARNINGS", None);
            }
            
            gpu_build.compile("gdeflate_gpu");
        }
        
        println!("cargo:rerun-if-changed=GDeflate/GDeflate_gpu.cpp");
        println!("cargo:rerun-if-changed=GDeflate/GDeflate_nvcomp.cpp");
        println!("cargo:rerun-if-changed=GDeflate/GDeflate_gpu.h");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=NVCOMP_ROOT");
    }

    // Link pthread on Unix
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=pthread");
    }

    // Re-run build script if sources change
    println!("cargo:rerun-if-changed=GDeflate/GDeflateCompress.cpp");
    println!("cargo:rerun-if-changed=GDeflate/GDeflateDecompress.cpp");
    println!("cargo:rerun-if-changed=GDeflate/GDeflate_c.cpp");
    println!("cargo:rerun-if-changed=GDeflate/GDeflate.h");
    println!("cargo:rerun-if-changed=GDeflate/GDeflate_c.h");
}
