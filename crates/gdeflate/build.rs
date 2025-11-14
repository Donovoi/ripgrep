use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct NvcompConfig {
    include_dirs: Vec<PathBuf>,
    lib_dirs: Vec<PathBuf>,
    libs: Vec<&'static str>,
}

fn add_if_thrust_present(dir: PathBuf, out: &mut Vec<PathBuf>) {
    if dir.join("thrust").join("device_vector.h").exists()
        && !out.iter().any(|existing| existing == &dir)
    {
        out.push(dir);
    }
}

fn detect_thrust_include(cuda_include: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    add_if_thrust_present(cuda_include.to_path_buf(), &mut dirs);

    if let Ok(thrust_root) = env::var("THRUST_ROOT") {
        let root_path = PathBuf::from(thrust_root);
        add_if_thrust_present(root_path.clone(), &mut dirs);
        add_if_thrust_present(root_path.join("include"), &mut dirs);
    }

    for candidate in [
        PathBuf::from("/usr/include"),
        PathBuf::from("/usr/local/include"),
        PathBuf::from("/opt/nvidia/thrust/include"),
    ] {
        add_if_thrust_present(candidate, &mut dirs);
    }

    dirs
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

#[cfg(not(target_os = "windows"))]
fn compiler_major_version(compiler: &std::ffi::OsStr) -> Option<u32> {
    let try_version = |arg: &str| {
        Command::new(compiler)
            .arg(arg)
            .output()
            .ok()
            .filter(|output| output.status.success())
    };

    let output = try_version("-dumpfullversion")
        .or_else(|| try_version("-dumpversion"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().split('.').next().and_then(|major| major.parse::<u32>().ok())
}

#[cfg(not(target_os = "windows"))]
fn resolve_executable(candidate: &str) -> Option<PathBuf> {
    let path = Path::new(candidate);
    if path.is_absolute() || path.components().count() > 1 {
        return path.exists().then_some(path.to_path_buf());
    }

    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let full_path = dir.join(candidate);
        if full_path.exists() {
            return Some(full_path);
        }
    }
    None
}

fn detect_cuda_host_compiler() -> Option<PathBuf> {
    if let Ok(host) = env::var("CUDAHOSTCXX") {
        let path = PathBuf::from(&host);
        if path.is_absolute() || path.components().count() > 1 {
            return Some(path);
        }
        #[cfg(not(target_os = "windows"))]
        if let Some(resolved) = resolve_executable(&host) {
            return Some(resolved);
        }
        return Some(PathBuf::from(host));
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(explicit) = env::var("RG_CUDA_HOST_COMPILER") {
            if let Some(path) = resolve_executable(&explicit) {
                return Some(path);
            } else {
                println!(
                    "cargo:warning=RG_CUDA_HOST_COMPILER was set to {} but the compiler was not found",
                    explicit
                );
            }
        }

        if let Some(major) =
            compiler_major_version(std::ffi::OsStr::new("c++"))
        {
            if major <= 12 {
                return None;
            }
        }

        let fallback_candidates = [
            "c++-12", "g++-12", "c++-11", "g++-11", "c++-10", "g++-10",
            "c++-9", "g++-9",
        ];

        for candidate in &fallback_candidates {
            if let Some(path) = resolve_executable(candidate) {
                if let Some(major) = compiler_major_version(path.as_os_str())
                    .or_else(|| {
                        compiler_major_version(std::ffi::OsStr::new(candidate))
                    })
                {
                    if major <= 12 {
                        return Some(path);
                    }
                }
            }
        }
    }

    None
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
                .file(gdeflate_src.join("GpuSubstringSearch.cu"))
                .cpp(true)
                .cuda(true)
                .include(&gdeflate_src)
                .include(&cuda_include)
                .define("CUDA_GPU_SUPPORT", None)
                .warnings(false);

            if cfg!(debug_assertions) {
                // The cc crate injects -G for debug builds when compiling CUDA
                // sources, which triggers a known NVCC stub generation bug
                // with heavily templatized Thrust kernels. Disable device
                // debug info explicitly while keeping host-side debug output.
                gpu_build.debug(false);
                gpu_build.opt_level(0);
                gpu_build.flag("-Xcompiler");
                gpu_build.flag("-Og");
                gpu_build.flag("-Xcompiler");
                gpu_build.flag("-g");
            }

            if let Some(host_compiler) = detect_cuda_host_compiler() {
                if let Some(path) = host_compiler.to_str() {
                    println!(
                        "cargo:warning=Using {} as CUDA host compiler (set CUDAHOSTCXX or RG_CUDA_HOST_COMPILER to override)",
                        path
                    );
                    gpu_build.flag("-ccbin");
                    gpu_build.flag(path);
                }
            }

            // Newer distributions often ship host compilers that are newer
            // than what the bundled NVCC officially supports. Rather than
            // fail with opaque parser errors when that happens, opt into
            // NVCC's forward-compatibility mode so that we can build with
            // the system toolchain by default.
            gpu_build.flag("-allow-unsupported-compiler");

            let float_compat = gdeflate_src.join("FloatCompat.h");
            if let Some(path) = float_compat.to_str() {
                gpu_build.flag("-include");
                gpu_build.flag(path);
            }

            gpu_build.flag("-std=c++17");
            gpu_build.flag("-Xcompiler");
            gpu_build.flag("-std=gnu++17");

            let thrust_includes = detect_thrust_include(&cuda_include);
            if thrust_includes.is_empty() {
                println!(
                    "cargo:warning=Thrust headers not found; install libthrust-dev or set THRUST_ROOT"
                );
            } else {
                for dir in &thrust_includes {
                    gpu_build.include(dir);
                }
            }

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
        println!("cargo:rerun-if-changed=GDeflate/GpuSubstringSearch.cu");
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
