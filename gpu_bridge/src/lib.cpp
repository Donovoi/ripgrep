#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include "gpu_search.cuh"

extern "C" {

struct RgGpuCompileOptions {
    bool case_sensitive;
    bool dotall;
    bool multiline;
    bool unicode;
};

struct RgGpuSearchInput {
    uint64_t file_len;
    bool stats_enabled;
    const char* path_ptr;
    size_t path_len;
};

struct RgGpuSearchStats {
    uint64_t elapsed_ns;
    uint64_t bytes_scanned;
};

struct RgGpuSearchResult {
    int32_t status;
    RgGpuSearchStats stats;
};

// Status codes
const int32_t STATUS_NO_MATCH = 0;
const int32_t STATUS_MATCH_FOUND = 1;
const int32_t STATUS_ERROR = -1;

// A simple struct to hold the compiled pattern
struct GpuRegexPattern {
    GpuPattern pattern;
};

int32_t rg_gpu_regex_compile(
    const uint8_t* pattern_ptr,
    size_t pattern_len,
    const RgGpuCompileOptions* options,
    void** out_handle
) {
    if (!pattern_ptr || !options || !out_handle) {
        return STATUS_ERROR;
    }

    try {
        std::string pat(reinterpret_cast<const char*>(pattern_ptr), pattern_len);
        
        // For testing purposes, we can reject patterns that start with "fail_compile"
        if (pat.find("fail_compile") == 0) {
            return STATUS_ERROR;
        }

        auto* compiled = new GpuRegexPattern{
            GpuPattern{
                pat,
                options->case_sensitive,
                options->dotall
            }
        };
        *out_handle = static_cast<void*>(compiled);
        return 0; // Success
    } catch (...) {
        return STATUS_ERROR;
    }
}

void rg_gpu_regex_release(void* handle) {
    if (handle) {
        delete static_cast<GpuRegexPattern*>(handle);
    }
}

int32_t rg_gpu_regex_search(
    void* handle,
    const RgGpuSearchInput* input,
    RgGpuSearchResult* result
) {
    if (!handle || !input || !result) {
        return STATUS_ERROR;
    }

    auto* compiled = static_cast<GpuRegexPattern*>(handle);
    
    // Construct path string
    std::string path(input->path_ptr, input->path_len);

    uint64_t elapsed_ns = 0;
    bool found = launch_gpu_search(
        compiled->pattern,
        path.c_str(),
        input->file_len,
        &elapsed_ns
    );

    result->status = found ? STATUS_MATCH_FOUND : STATUS_NO_MATCH;
    result->stats.elapsed_ns = elapsed_ns;
    result->stats.bytes_scanned = input->file_len;

    return 0; // Success
}

} // extern "C"
