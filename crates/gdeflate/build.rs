use std::env;
use std::fs;
use std::path::{Path, PathBuf};

struct NvcompConfig {
    include_dirs: Vec<PathBuf>,
    lib_dirs: Vec<PathBuf>,
    libs: Vec<&'static str>,
}

fn library_exists(dir: &Path, lib_name: &str) -> bool {
    if !dir.is_dir() {
        return false;
    }

    let prefix = format!("lib{}", lib_name);
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str())
            {
                if file_name.starts_with(&prefix)
                    && (file_name.ends_with(".so")
                        || file_name.contains(".so.")
                        || file_name.ends_with(".a"))
                {
                    return true;
                }
            }
        }
    }

    false
}

fn collect_nvcomp_lib_dirs(candidate: &Path, out: &mut Vec<PathBuf>) {
    if !candidate.is_dir() {
        return;
    }

    if library_exists(candidate, "nvcomp")
        && out.iter().all(|dir| dir != candidate)
    {
        out.push(candidate.to_path_buf());
    }

    let nested = candidate.join("nvcomp");
    if nested.is_dir() {
        if library_exists(&nested, "nvcomp")
            && out.iter().all(|dir| dir != &nested)
        {
            out.push(nested.clone());
        }

        if let Ok(entries) = fs::read_dir(&nested) {
            for entry in entries.flatten() {
                let subdir = entry.path();
                if subdir.is_dir()
                    && library_exists(&subdir, "nvcomp")
                    && out.iter().all(|dir| dir != &subdir)
                {
                    out.push(subdir);
                }
            }
        }
    }

    let stubs = candidate.join("stubs");
    if stubs.is_dir()
        && library_exists(&stubs, "nvcomp")
        && out.iter().all(|dir| dir != &stubs)
    {
        out.push(stubs);
    }
}

fn header_parent_if_exists(path: PathBuf) -> Option<PathBuf> {
    if path.exists() {
        return path.parent().map(|p| p.to_path_buf());
    }
    None
}

fn probe_nvcomp_root(root: &Path) -> Option<NvcompConfig> {
    let mut include_dirs = Vec::new();

    let include_base = root.join("include");
    let include_candidates = [
        include_base.join("nvcomp.h"),
        include_base.join("nvcomp").join("nvcomp.h"),
        include_base.join("nvcomp_14").join("nvcomp.h"),
        include_base.join("nvcomp_13").join("nvcomp.h"),
        include_base.join("nvcomp_12").join("nvcomp.h"),
        include_base.join("nvcomp_11").join("nvcomp.h"),
    ];

    for candidate in include_candidates {
        if let Some(parent) = header_parent_if_exists(candidate) {
            include_dirs.push(parent);
            break;
        }
    }

    if include_dirs.is_empty() {
        return None;
    }

    let mut lib_dirs = Vec::new();
    let lib_candidates = [
        root.join("lib"),
        root.join("lib64"),
        root.join("lib/x86_64-linux-gnu"),
        root.join("lib/aarch64-linux-gnu"),
        root.join("lib64/x86_64-linux-gnu"),
        root.join("lib64/aarch64-linux-gnu"),
    ];

    for candidate in &lib_candidates {
        collect_nvcomp_lib_dirs(candidate, &mut lib_dirs);
    }

    if root == Path::new("/usr") {
        collect_nvcomp_lib_dirs(
            Path::new("/usr/lib/x86_64-linux-gnu"),
            &mut lib_dirs,
        );
        collect_nvcomp_lib_dirs(
            Path::new("/usr/lib/aarch64-linux-gnu"),
            &mut lib_dirs,
        );
    }

    lib_dirs.sort();
    lib_dirs.dedup();

    if lib_dirs.is_empty() {
        return None;
    }

    let mut libs: Vec<&'static str> = vec!["nvcomp"];
    if lib_dirs.iter().any(|dir| library_exists(dir, "nvcomp_gdeflate")) {
        libs.push("nvcomp_gdeflate");
    }

    Some(NvcompConfig { include_dirs, lib_dirs, libs })
}

fn detect_nvcomp(cuda_path: &Path) -> Option<NvcompConfig> {
    if let Ok(nvcomp_root) = env::var("NVCOMP_ROOT") {
        let root_path = Path::new(&nvcomp_root);
        if let Some(config) = probe_nvcomp_root(root_path) {
            println!(
                "cargo:warning=Found nvCOMP at NVCOMP_ROOT: {}",
                nvcomp_root
            );
            return Some(config);
        } else {
            println!(
                "cargo:warning=NVCOMP_ROOT was set to {} but nvCOMP headers or libraries were not found",
                nvcomp_root
            );
        }
    }

    if let Some(config) = probe_nvcomp_root(cuda_path) {
        println!("cargo:warning=Found nvCOMP in CUDA installation");
        return Some(config);
    }

    for system_root in [
        Path::new("/usr/local"),
        Path::new("/usr"),
        Path::new("/opt/nvidia"),
        Path::new("/opt/nvcomp"),
    ] {
        if let Some(config) = probe_nvcomp_root(system_root) {
            println!(
                "cargo:warning=Found nvCOMP at system path: {}",
                system_root.display()
            );
            return Some(config);
        }
    }

    None
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
            let nvcomp_config = detect_nvcomp(Path::new(&cuda_path));

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

            if let Some(config) = &nvcomp_config {
                println!("cargo:warning=nvCOMP library found, enabling GPU decompression");
                gpu_build.define("NVCOMP_AVAILABLE", None);

                // Add nvCOMP source file
                gpu_build.file(gdeflate_src.join("GDeflate_nvcomp.cpp"));

                for include_dir in &config.include_dirs {
                    gpu_build.include(include_dir);
                }
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
            if let Some(config) = nvcomp_config {
                for dir in config.lib_dirs {
                    println!(
                        "cargo:rustc-link-search=native={}",
                        dir.display()
                    );
                }
                for lib in config.libs {
                    println!("cargo:rustc-link-lib={}", lib);
                }
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
