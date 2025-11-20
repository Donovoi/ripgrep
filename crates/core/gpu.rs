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

    use std::{ffi::c_void, ptr::NonNull, time::Duration};

    use anyhow::{Context, bail};
    use regex_automata::dfa::{Automaton, dense};

    unsafe extern "C" {
        fn rg_gpu_regex_compile(
            pattern_ptr: *const u8,
            pattern_len: usize,
            options: *const RgGpuCompileOptions,
            out_handle: *mut *mut c_void,
        ) -> i32;

        fn rg_gpu_regex_release(handle: *mut c_void);

        fn rg_gpu_regex_search(
            handle: *mut c_void,
            input: *const RgGpuSearchInput,
            result: *mut RgGpuSearchResult,
        ) -> i32;
    }

    #[derive(Debug, Clone)]
    pub(super) struct NvtextRegexEngine {
        stats_enabled: bool,
    }

    impl NvtextRegexEngine {
        pub(super) fn new(stats_enabled: bool) -> anyhow::Result<Self> {
            Ok(Self { stats_enabled })
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
            // Build DFA using regex-automata
            // We use Unanchored to allow finding matches anywhere in the text.
            // This adds a loop on the start state for any byte.
            let dfa =
                dense::Builder::new()
                    .configure(dense::Config::new().start_kind(
                        regex_automata::dfa::StartKind::Unanchored,
                    ))
                    .syntax(
                        regex_automata::util::syntax::Config::new()
                            .case_insensitive(!request.case_sensitive)
                            .dot_matches_new_line(request.dotall)
                            .unicode(request.unicode)
                            .multi_line(request.multiline),
                    )
                    .build(request.pattern)
                    .context("failed to compile regex to DFA")?;

            // Serialize DFA table
            // We need a flat table: [state * 256 + byte] -> next_state
            // And a way to identify match states.
            // For simplicity, we'll use a u32 array where the high bit indicates match.
            // Or we can just pass the raw bytes if we can decode it on GPU.
            // But decoding the dense::DFA format on GPU is complex.
            // Let's build a simple flat table.

            // Iterate over all states
            // Note: regex-automata states are IDs.
            // We need to map them to 0..N
            // Actually, dfa.next_state(id, byte) returns the next ID.

            // We will just dump the raw transition table if possible, but the ID space might be sparse or special.
            // Let's just iterate 0..state_count? No, state IDs are not necessarily contiguous 0..N in that way?
            // Actually dense::DFA uses contiguous IDs.

            // Let's create a simple mapping.
            // We need to know the start state.
            let start_state = dfa
                .start_state_forward(&regex_automata::Input::new(b""))
                .unwrap();

            // We will use a flat vector of u32s.
            // Index = (state_id * 256) + byte
            // Value = next_state_id | (is_match << 31)

            // Wait, state IDs in regex-automata are multiples of stride?
            // Let's check docs or assume standard behavior.
            // In dense::DFA, state IDs are byte offsets into the transition table.
            // So state_id / stride is the logical index.

            // Let's just use the state IDs directly as provided by the DFA.
            // But we need to know the max ID to allocate the GPU buffer.
            // dfa.memory_usage() gives bytes.

            // Let's construct a simplified table for the GPU.
            // We'll map every reachable state to a dense integer 0..N.
            // But to avoid re-traversing, let's just use the existing structure if we can.
            // The issue is that `dfa.next_state(state, byte)` is fast.

            // Let's just build a flat table of size (num_states * 256).
            // We need to iterate all valid state IDs.
            // regex-automata doesn't easily expose "all state IDs" without walking.
            // But since it's a dense DFA, we can assume states are somewhat contiguous or we can walk it.

            // Actually, let's just use `to_bytes_little_endian` and pass the whole blob?
            // Then the GPU kernel would need to understand the format.
            // The format is: header, then transition table.
            // The transition table is `state_id + byte`.
            // This is exactly what we want!
            // The only issue is endianness (GPU is usually LE, same as x86).
            // And the specific layout.

            // Let's try to use the raw bytes.
            let (_bytes, _) = dfa.to_bytes_little_endian();

            // We need to pass this buffer to the C++ bridge.
            // We also need to tell the bridge where the start state is.
            // And how to detect a match.
            // In dense::DFA, match states are those where `dfa.is_match_state(id)` is true.
            // This information is encoded in the state ID or a separate table?
            // In dense::DFA, it's usually a separate check or encoded in the ID.
            // Actually, `is_match_state` checks if the state ID is a "special" state.

            // If we use the raw bytes, we need to reimplement `next_state` and `is_match_state` logic on GPU.
            // That might be brittle if internal format changes.

            // Alternative: Build our own simple table.
            // Walk the DFA from start state.
            let mut stack = vec![start_state];
            let mut visited = std::collections::HashMap::new();
            let mut ordered_states = Vec::new();

            visited.insert(start_state, 0u32);
            ordered_states.push(start_state);

            let mut i = 0;
            while i < ordered_states.len() {
                let current_id = ordered_states[i];
                i += 1;

                for b in 0..=255 {
                    let next_id = dfa.next_state(current_id, b);
                    if !visited.contains_key(&next_id) {
                        let new_idx = visited.len() as u32;
                        visited.insert(next_id, new_idx);
                        ordered_states.push(next_id);
                    }
                }
            }

            // Now build the flat table
            // table[current_idx * 256 + b] = next_idx
            // And a bitset for match states? Or encode in the value?
            // Let's use u32. High bit = match.
            let mut flat_table = vec![0u32; ordered_states.len() * 256];
            for (idx, &original_id) in ordered_states.iter().enumerate() {
                let _is_match = dfa.is_match_state(original_id);
                for b in 0..=255 {
                    let next_id = dfa.next_state(original_id, b as u8);
                    let next_idx = visited[&next_id];
                    let mut val = next_idx;
                    // If the *next* state is a match, we might want to know immediately?
                    // Or we check if *current* state is match?
                    // Usually we transition, then check if new state is match.
                    // So we need to know if `next_idx` is a match state.
                    // But `is_match` is a property of the state.
                    // So let's store `is_match` in the table entry?
                    // No, `is_match` is a property of the *target* state.
                    // So when we look up `table[curr][b]`, we get `next`.
                    // We can encode `is_match(next)` in the high bit of `next`.

                    if dfa.is_match_state(next_id) {
                        val |= 0x8000_0000;
                    }
                    flat_table[idx * 256 + b] = val;
                }
            }

            let options = RgGpuCompileOptions {
                case_sensitive: request.case_sensitive,
                dotall: request.dotall,
                multiline: request.multiline,
                unicode: request.unicode,
            };

            // We need to pass the table to the bridge.
            // We can pass it as the "pattern" pointer, or add a new field.
            // Since we are changing the protocol, let's just cast the table pointer.
            // But `compile` takes `pattern_ptr` and `pattern_len`.
            // We can pass the table as bytes.

            let table_bytes_len = flat_table.len() * 4;
            let table_ptr = flat_table.as_ptr() as *const u8;

            let mut handle: *mut c_void = std::ptr::null_mut();
            let status = unsafe {
                rg_gpu_regex_compile(
                    table_ptr,
                    table_bytes_len,
                    &options,
                    &mut handle,
                )
            };
            if status != 0 {
                bail!("nvtext bridge rejected pattern (status {status})");
            }
            let handle = NonNull::new(handle)
                .context("nvtext bridge returned null regex handle")?;
            Ok(NvtextProgram { handle })
        }

        fn search_path(
            &self,
            program: &Self::Program,
            input: &GpuRegexInput<'_>,
        ) -> anyhow::Result<GpuRegexSearchOutcome> {
            // Heuristic: Only use GPU for files larger than 2MB.
            // Smaller files are faster on CPU due to transfer overhead.
            const MIN_GPU_FILE_SIZE: u64 = 2 * 1024 * 1024;
            // If file_len is 0, it might be a block device, so we proceed.
            if input.file_len > 0 && input.file_len < MIN_GPU_FILE_SIZE {
                return Ok(GpuRegexSearchOutcome::NotAttempted);
            }

            // Read file content
            // Note: We read the file here in Rust to pass the buffer to C++.
            // This allows future optimizations like batching or memory mapping.
            let mut file = std::fs::File::open(input.path)
                .context("failed to open file for GPU search")?;

            // If file_len is 0, try to determine size via seek (for block devices)
            let mut len = input.file_len;
            if len == 0 {
                use std::io::Seek;
                len = file
                    .seek(std::io::SeekFrom::End(0))
                    .context("failed to seek to end of block device")?;
                // We don't need to seek back for mmap usually, but let's be safe
                // file.seek(std::io::SeekFrom::Start(0))?;
            }

            if len == 0 {
                return Ok(GpuRegexSearchOutcome::NoMatch(
                    GpuRegexExecutionStats::default(),
                ));
            }

            // For block devices, we must use the determined length for mmap
            let mmap = unsafe {
                memmap2::MmapOptions::new().len(len as usize).map(&file)
            }
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
                data_len: len,
                stats_enabled: input.stats_enabled,
                data_ptr: mmap.as_ptr(),
            };
            let status = unsafe {
                rg_gpu_regex_search(
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

    pub(super) struct NvtextProgram {
        handle: NonNull<c_void>,
    }

    unsafe impl Send for NvtextProgram {}
    unsafe impl Sync for NvtextProgram {}

    impl Drop for NvtextProgram {
        fn drop(&mut self) {
            unsafe { rg_gpu_regex_release(self.handle.as_ptr()) };
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
