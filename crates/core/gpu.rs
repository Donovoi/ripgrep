#![allow(dead_code)]

//! GPU regex abstraction layer.
//!
//! This module intentionally contains forward-looking interfaces that will be
//! exercised incrementally as we integrate a real CUDA-backed regex engine. The
//! items are `pub(crate)` so other modules can start wiring them up without
//! immediately enabling GPU execution.

use std::sync::Arc;
use std::{path::Path, time::Duration};

/// Describes the high-level regex features a compiled GPU program must honor.
#[derive(Clone, Debug)]
pub(crate) struct GpuRegexCompileRequest<'a> {
    /// The raw pattern string exactly as provided by the user (after any
    /// canonicalization performed by ripgrep's parser).
    pub pattern: &'a str,
    /// Whether the match semantics should be case-sensitive.
    pub case_sensitive: bool,
    /// Whether `.` should match newlines.
    pub dotall: bool,
    /// Whether multi-line mode was requested (i.e., `^`/`$` match line
    /// boundaries within the text).
    pub multiline: bool,
    /// Whether Unicode-aware character classes are required.
    pub unicode: bool,
}

impl<'a> GpuRegexCompileRequest<'a> {
    /// Convenience constructor for the most common configuration (case
    /// sensitive, single-line, ASCII mode).
    pub(crate) fn literal(pattern: &'a str) -> Self {
        Self {
            pattern,
            case_sensitive: true,
            dotall: false,
            multiline: false,
            unicode: false,
        }
    }
}

/// Metadata describing the file (or stream) being evaluated on the GPU.
#[derive(Clone, Debug)]
pub(crate) struct GpuRegexInput<'a> {
    pub path: &'a Path,
    pub file_len: u64,
    pub stats_enabled: bool,
}

/// Aggregate statistics gathered from a completed GPU regex pass.
#[derive(Clone, Debug, Default)]
pub(crate) struct GpuRegexExecutionStats {
    pub elapsed: Duration,
    pub bytes_scanned: u64,
}

/// Result of attempting to execute a compiled GPU regex against some input.
#[derive(Clone, Debug)]
pub(crate) enum GpuRegexSearchOutcome {
    /// GPU execution was skipped (engine unavailable, unsupported feature,
    /// file too small, etc.). Callers should fall back to the CPU engine.
    NotAttempted,
    /// GPU execution succeeded and determined that no matches exist.
    NoMatch(GpuRegexExecutionStats),
    /// GPU execution succeeded and detected at least one match.
    MatchFound(GpuRegexExecutionStats),
}

/// Abstraction over any CUDA-capable regex engine.
pub(crate) trait GpuRegexEngine: Send + Sync + 'static {
    /// Handle returned by `compile` and consumed by `search_path`.
    type Program: Send + Sync;

    /// Human-friendly identifier for logging/diagnostics.
    fn name(&self) -> &'static str;

    /// Compile the supplied pattern into a GPU-ready program.
    fn compile(
        &self,
        request: &GpuRegexCompileRequest<'_>,
    ) -> anyhow::Result<Self::Program>;

    /// Execute a compiled program against the specified input.
    fn search_path(
        &self,
        program: &Self::Program,
        input: &GpuRegexInput<'_>,
    ) -> anyhow::Result<GpuRegexSearchOutcome>;
}

/// Placeholder engine used when CUDA support is not enabled or not yet wired.
#[derive(Debug, Default)]
pub(crate) struct GpuRegexStubEngine;

impl GpuRegexEngine for GpuRegexStubEngine {
    type Program = ();

    fn name(&self) -> &'static str {
        "gpu-stub"
    }

    fn compile(
        &self,
        _request: &GpuRegexCompileRequest<'_>,
    ) -> anyhow::Result<Self::Program> {
        Ok(())
    }

    fn search_path(
        &self,
        _program: &Self::Program,
        _input: &GpuRegexInput<'_>,
    ) -> anyhow::Result<GpuRegexSearchOutcome> {
        Ok(GpuRegexSearchOutcome::NotAttempted)
    }
}

/// Erased executable returned to higher layers once a pattern is compiled.
pub(crate) trait GpuRegexExecutable: Send + Sync {
    fn engine_name(&self) -> &'static str;
    fn search_path(
        &self,
        input: &GpuRegexInput<'_>,
    ) -> anyhow::Result<GpuRegexSearchOutcome>;
}

struct EngineProgram<E: GpuRegexEngine> {
    engine: Arc<E>,
    program: E::Program,
}

impl<E: GpuRegexEngine> EngineProgram<E> {
    fn new(engine: Arc<E>, program: E::Program) -> Self {
        Self { engine, program }
    }
}

impl<E: GpuRegexEngine> GpuRegexExecutable for EngineProgram<E> {
    fn engine_name(&self) -> &'static str {
        self.engine.name()
    }

    fn search_path(
        &self,
        input: &GpuRegexInput<'_>,
    ) -> anyhow::Result<GpuRegexSearchOutcome> {
        self.engine.search_path(&self.program, input)
    }
}

fn compile_with_engine<E: GpuRegexEngine>(
    engine: Arc<E>,
    request: &GpuRegexCompileRequest<'_>,
) -> Option<Box<dyn GpuRegexExecutable>> {
    match engine.compile(request) {
        Ok(program) => {
            log::debug!(
                "GPU regex engine '{}' compiled pattern",
                engine.name()
            );
            Some(Box::new(EngineProgram::new(engine, program)))
        }
        Err(err) => {
            log::debug!(
                "GPU regex engine '{}' unavailable: {}",
                engine.name(),
                err
            );
            None
        }
    }
}

#[allow(dead_code)]
pub(crate) fn try_compile_default_engine(
    request: &GpuRegexCompileRequest<'_>,
    stats_enabled: bool,
) -> Option<Box<dyn GpuRegexExecutable>> {
    #[cfg(feature = "cuda-gpu")]
    {
        if let Some(exec) = nvtext::try_compile(request, stats_enabled) {
            return Some(exec);
        }
    }

    let _ = stats_enabled;

    compile_with_engine(Arc::new(GpuRegexStubEngine::default()), request)
}

#[cfg(feature = "cuda-gpu")]
mod nvtext {
    use super::{
        Arc, GpuRegexCompileRequest, GpuRegexEngine, GpuRegexExecutable,
        GpuRegexExecutionStats, GpuRegexInput, GpuRegexSearchOutcome,
        compile_with_engine,
    };

    use std::{env, ffi::c_void, ptr::NonNull, time::Duration};

    use anyhow::{Context, bail};
    use libloading::Library;

    const DEFAULT_SYMBOL_COMPILE: &[u8] = b"rg_gpu_regex_compile\0";
    const DEFAULT_SYMBOL_RELEASE: &[u8] = b"rg_gpu_regex_release\0";
    const DEFAULT_SYMBOL_SEARCH: &[u8] = b"rg_gpu_regex_search\0";

    #[derive(Debug, Clone)]
    pub(super) struct NvtextRegexEngine {
        bridge: Arc<NvtextLibrary>,
        stats_enabled: bool,
    }

    impl NvtextRegexEngine {
        pub(super) fn new(stats_enabled: bool) -> anyhow::Result<Self> {
            let bridge = NvtextLibrary::load()?;
            Ok(Self { bridge, stats_enabled })
        }
    }

    impl GpuRegexEngine for NvtextRegexEngine {
        type Program = NvtextProgram;

        fn name(&self) -> &'static str {
            "nvtext"
        }

        fn compile(
            &self,
            request: &GpuRegexCompileRequest<'_>,
        ) -> anyhow::Result<Self::Program> {
            let options = RgGpuCompileOptions {
                case_sensitive: request.case_sensitive,
                dotall: request.dotall,
                multiline: request.multiline,
                unicode: request.unicode,
            };
            let mut handle: *mut c_void = std::ptr::null_mut();
            let status = unsafe {
                (self.bridge.compile)(
                    request.pattern.as_ptr(),
                    request.pattern.len(),
                    &options,
                    &mut handle,
                )
            };
            if status != 0 {
                bail!("nvtext bridge rejected pattern (status {status})");
            }
            let handle = NonNull::new(handle)
                .context("nvtext bridge returned null regex handle")?;
            Ok(NvtextProgram { bridge: Arc::clone(&self.bridge), handle })
        }

        fn search_path(
            &self,
            program: &Self::Program,
            input: &GpuRegexInput<'_>,
        ) -> anyhow::Result<GpuRegexSearchOutcome> {
            // Read file content
            // Note: We read the file here in Rust to pass the buffer to C++.
            // This allows future optimizations like batching or memory mapping.
            let file = std::fs::File::open(input.path)
                .context("failed to open file for GPU search")?;

            // If the file is empty, we can skip it.
            if input.file_len == 0 {
                return Ok(GpuRegexSearchOutcome::NoMatch(
                    GpuRegexExecutionStats::default(),
                ));
            }

            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .context("failed to mmap file for GPU search")?;

            // Allocate buffer for matches
            let max_matches = 4096;
            let mut matches = vec![RgGpuMatch::default(); max_matches];

            let mut result = RgGpuSearchResult {
                matches: matches.as_mut_ptr(),
                max_matches,
                ..Default::default()
            };

            let request = RgGpuSearchInput {
                data_len: mmap.len() as u64,
                stats_enabled: input.stats_enabled,
                data_ptr: mmap.as_ptr(),
            };
            let status = unsafe {
                (program.bridge.search)(
                    program.handle.as_ptr(),
                    &request,
                    &mut result,
                )
            };
            if status != 0 {
                bail!("nvtext search failed (status {status})");
            }
            Ok(result.into_outcome())
        }
    }

    pub(super) fn try_compile(
        request: &GpuRegexCompileRequest<'_>,
        stats_enabled: bool,
    ) -> Option<Box<dyn GpuRegexExecutable>> {
        match NvtextRegexEngine::new(stats_enabled) {
            Ok(engine) => compile_with_engine(Arc::new(engine), request),
            Err(err) => {
                log::debug!("nvtext engine unavailable: {err}");
                None
            }
        }
    }

    #[derive(Debug)]
    pub(super) struct NvtextLibrary {
        _lib: Library,
        compile: CompileFn,
        release: ReleaseFn,
        search: SearchFn,
    }

    type CompileFn = unsafe extern "C" fn(
        pattern_ptr: *const u8,
        pattern_len: usize,
        options: *const RgGpuCompileOptions,
        out_handle: *mut *mut c_void,
    ) -> i32;
    type ReleaseFn = unsafe extern "C" fn(handle: *mut c_void);
    type SearchFn = unsafe extern "C" fn(
        handle: *mut c_void,
        input: *const RgGpuSearchInput,
        result: *mut RgGpuSearchResult,
    ) -> i32;

    impl NvtextLibrary {
        fn load() -> anyhow::Result<Arc<Self>> {
            let mut errors = Vec::new();
            if let Ok(explicit) = env::var("RG_NVTEXT_BRIDGE_PATH") {
                match Self::open(&explicit) {
                    Ok(lib) => return Ok(Arc::new(lib)),
                    Err(err) => errors.push(format!("{}: {err}", explicit)),
                }
            }
            for candidate in default_library_candidates() {
                match Self::open(candidate) {
                    Ok(lib) => return Ok(Arc::new(lib)),
                    Err(err) => errors.push(format!("{}: {err}", candidate)),
                }
            }
            bail!(
                "failed to load nvtext GPU bridge. attempted: {}",
                errors.join(", ")
            );
        }

        fn open(path: &str) -> anyhow::Result<Self> {
            unsafe {
                let lib = Library::new(path)?;
                let compile = *lib.get::<CompileFn>(DEFAULT_SYMBOL_COMPILE)?;
                let release = *lib.get::<ReleaseFn>(DEFAULT_SYMBOL_RELEASE)?;
                let search = *lib.get::<SearchFn>(DEFAULT_SYMBOL_SEARCH)?;
                Ok(Self { _lib: lib, compile, release, search })
            }
        }
    }

    fn default_library_candidates() -> &'static [&'static str] {
        #[cfg(target_os = "windows")]
        {
            &["rg_gpu_regex_bridge.dll"]
        }
        #[cfg(target_os = "macos")]
        {
            &["librg_gpu_regex_bridge.dylib", "libnvtext_rg_bridge.dylib"]
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            &["librg_gpu_regex_bridge.so", "libnvtext_rg_bridge.so"]
        }
    }

    pub(super) struct NvtextProgram {
        bridge: Arc<NvtextLibrary>,
        handle: NonNull<c_void>,
    }

    unsafe impl Send for NvtextProgram {}
    unsafe impl Sync for NvtextProgram {}

    impl Drop for NvtextProgram {
        fn drop(&mut self) {
            unsafe { (self.bridge.release)(self.handle.as_ptr()) };
        }
    }

    #[repr(C)]
    struct RgGpuCompileOptions {
        case_sensitive: bool,
        dotall: bool,
        multiline: bool,
        unicode: bool,
    }

    #[repr(C)]
    struct RgGpuSearchInput {
        data_len: u64,
        stats_enabled: bool,
        data_ptr: *const u8,
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct RgGpuSearchStats {
        elapsed_ns: u64,
        bytes_scanned: u64,
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct RgGpuMatch {
        offset: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct RgGpuSearchResult {
        status: i32,
        stats: RgGpuSearchStats,
        matches: *mut RgGpuMatch,
        match_count: usize,
        max_matches: usize,
    }

    impl RgGpuSearchResult {
        fn into_outcome(self) -> GpuRegexSearchOutcome {
            match self.status {
                STATUS_MATCH_FOUND => {
                    // We need to copy the matches from the C buffer if we want to use them
                    // But for now, GpuRegexSearchOutcome only supports stats.
                    // We should update GpuRegexSearchOutcome to support offsets.
                    GpuRegexSearchOutcome::MatchFound(self.into_stats())
                }
                STATUS_NO_MATCH => {
                    GpuRegexSearchOutcome::NoMatch(self.into_stats())
                }
                _ => GpuRegexSearchOutcome::NotAttempted,
            }
        }

        fn into_stats(self) -> GpuRegexExecutionStats {
            GpuRegexExecutionStats {
                elapsed: Duration::from_nanos(self.stats.elapsed_ns),
                bytes_scanned: self.stats.bytes_scanned,
            }
        }
    }

    const STATUS_NO_MATCH: i32 = 0;
    const STATUS_MATCH_FOUND: i32 = 1;
}
