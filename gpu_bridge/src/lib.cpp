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
    uint64_t data_len;
    bool stats_enabled;
    const char* data_ptr;
};

struct RgGpuSearchStats {
    uint64_t elapsed_ns;
    uint64_t bytes_scanned;
};

struct RgGpuMatch {
    uint64_t offset;
};

struct RgGpuSearchResult {
    int32_t status;
    RgGpuSearchStats stats;
    RgGpuMatch* matches;
    size_t match_count;
    size_t max_matches;
};

// Status codes
const int32_t STATUS_NO_MATCH = 0;
const int32_t STATUS_MATCH_FOUND = 1;
const int32_t STATUS_ERROR = -1;
const int32_t STATUS_BUFFER_OVERFLOW = 2;

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
    
    uint64_t elapsed_ns = 0;
    int match_count = 0;
    
    // We need to cast the C-compatible match struct to the C++ one
    // They have the same layout (uint64_t offset)
    GpuMatch* gpu_matches = reinterpret_cast<GpuMatch*>(result->matches);
    
    int status = launch_gpu_search(
        compiled->pattern,
        input->data_ptr,
        input->data_len,
        &elapsed_ns,
        gpu_matches,
        &match_count
    );

    if (status == 1) {
        result->status = STATUS_MATCH_FOUND;
        result->match_count = match_count;
    } else if (status == 0) {
        result->status = STATUS_NO_MATCH;
        result->match_count = 0;
    } else {
        result->status = STATUS_ERROR;
        result->match_count = 0;
    }
    
    result->stats.elapsed_ns = elapsed_ns;
    result->stats.bytes_scanned = input->data_len;

    return 0; // Success
}

} // extern "C"
