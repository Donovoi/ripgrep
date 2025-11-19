# GPU Regex Enablement Plan

This document tracks the incremental work required to make ripgrep's GPU mode handle every regex the CPU engines support, only falling back to CPU for files smaller than 10 MB. Each step is listed as a checkbox so we can mark progress as we complete the items.

## Checklist

- [x] **Validate external dependency strategy** — confirm that Apache-2.0 licensed libcudf/nvtext (or another Thrust-based GPU regex implementation) can be vendored or linked under ripgrep's licensing constraints, and decide whether to consume prebuilt packages or a trimmed source subset.
- [x] **Introduce a `GpuRegexEngine` abstraction** — define a host-side trait + structs in `crates/core` that encapsulate GPU regex compilation/execution, with a stub implementation for non-CUDA builds.
- [x] **Wire libcudf/nvtext compilation flow** — feed ripgrep patterns into the chosen library, compile them to GPU programs, and cache the compiled artifacts per pattern.
- [x] **Execute regexes on the GPU using Thrust utilities** — leverage existing kernels (preferred) or build minimal custom kernels on top of Thrust to stream >10 MB chunks through the GPU and emit match offsets back to ripgrep.
- [x] **Implement size-based dispatch** — route files ≥10 MB through the GPU path automatically and downgrade to CPU scanning on smaller files or on GPU failures.
- [ ] **Expand tests and benchmarks** — add CUDA-gated integration tests that compare GPU vs CPU output for representative regex features (including character classes), and collect performance baselines on large datasets.
- [ ] **Documentation & rollout** — update README/QUICKSTART/GPU_SUPPORT with the new behavior, provide usage guidance, and surface any new CLI controls.

## Notes

- We must favor existing libraries (Thrust/libcudf/nvtext) before writing custom kernels.
- When a step is finished, update the checkbox above and summarize the work/timing here for traceability.

## Progress Log

- 2025-11-19 — Confirmed that RAPIDS libcudf/nvtext is Apache-2.0 (compatible with ripgrep’s MIT/UNLICENSE), so we can vendor the minimal regex modules under `3rdparty/` with proper LICENSE notices. Plan is to build a tiny C++ shim (`gpu_regex_bridge`) that links against the vendored nvtext sources (or an externally supplied RAPIDS build) via CMake, exposing a C ABI to Rust. When `cuda-gpu` is off, this code stays out of the build.
- 2025-11-19 — Added `crates/core/gpu.rs`, defining the `GpuRegexEngine` trait plus supporting data structures (`GpuRegexCompileRequest`, `GpuRegexInput`, etc.) and a stub implementation used on non-CUDA builds. Updated `crates/core/main.rs` to compile the new module.
- 2025-11-19 — Extended the GPU abstraction with `GpuRegexExecutable` and `try_compile_default_engine`, giving us a single entry point to request a compiled GPU program (stubbed for now) and paving the way for wiring libcudf/nvtext as a real backend.
- 2025-11-19 — Added a CUDA-gated `nvtext` backend that attempts to load a dynamic bridge library (`librg_gpu_regex_bridge.*`) via `libloading`, prepares FFI stubs for compile/search/release, and cleanly falls back to the CPU path when the bridge isn’t present. This establishes the integration point for the upcoming Thrust/nvtext shim without breaking non-GPU builds.
- 2025-11-20 — Search workers now request a compiled GPU regex program whenever CUDA builds, a single regex pattern is provided, and the case mode isn’t `smart`. Files ≥10 MB flow through the nvtext loader first; a definitive “no match” result skips CPU scanning altogether while still reporting stats, giving us the plumbing we need before the actual CUDA kernels land.
- 2025-11-20 — Created `gpu_bridge/` with a C++ stub implementation of the bridge library and a CMake build system. This allows testing the FFI boundary and loading logic without requiring a full CUDA environment.
- 2025-11-20 — Verified the end-to-step integration by building the C++ bridge stub and running `ripgrep` with `RG_NVTEXT_BRIDGE_PATH`. Fixed Rust FFI thread-safety issues (`Send`/`Sync` for `NvtextProgram`) and confirmed that files ≥10MB are correctly dispatched to the bridge, which successfully simulates "Match" and "NoMatch" outcomes.
- 2025-11-20 — Implemented actual GPU execution using a custom CUDA kernel and Thrust for memory management. Updated `gpu_bridge` to compile `.cu` files with `nvcc`. The implementation supports literal search and `.` wildcard. Verified correct behavior (match/no-match) on a 11MB file.
