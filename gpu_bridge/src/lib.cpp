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
    GpuDfa dfa;
};

int32_t rg_gpu_regex_compile(
    const uint8_t* pattern_ptr,
    size_t pattern_len,
    const RgGpuCompileOptions* options,
    void** out_handle
) {
    // Validate input parameters
    if (!pattern_ptr || !options || !out_handle) {
        return STATUS_ERROR;
    }

    try {
        // pattern_ptr now points to the DFA table (u32 array)
        // pattern_len is the size in bytes
        
        // Check alignment - pattern_ptr must be 4-byte aligned for uint32_t access
        if (reinterpret_cast<uintptr_t>(pattern_ptr) % alignof(uint32_t) != 0) {
            return STATUS_ERROR;
        }
        
        // Check size is multiple of uint32_t
        if (pattern_len % sizeof(uint32_t) != 0) {
            return STATUS_ERROR;
        }
        
        // Calculate number of entries
        size_t count = pattern_len / sizeof(uint32_t);
        
        // Validate reasonable DFA table size (max 10 million entries = ~40MB)
        // This prevents excessive memory allocation from malformed input
        const size_t MAX_DFA_TABLE_SIZE = 10 * 1024 * 1024;
        if (count == 0 || count > MAX_DFA_TABLE_SIZE) {
            return STATUS_ERROR;
        }
        
        const uint32_t* data = reinterpret_cast<const uint32_t*>(pattern_ptr);
        
        // Copy data into vector with bounds checking
        std::vector<uint32_t> table;
        table.reserve(count);
        try {
            table.assign(data, data + count);
        } catch (const std::bad_alloc&) {
            // Out of memory
            return STATUS_ERROR;
        }

        auto* compiled = new GpuRegexPattern{
            GpuDfa{
                std::move(table)
            }
        };
        *out_handle = static_cast<void*>(compiled);
        return 0; // Success
    } catch (const std::exception&) {
        return STATUS_ERROR;
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
        compiled->dfa,
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

// Stub implementation of launch_gpu_search when CUDA is not available
#ifndef CUDA_GPU_SUPPORT
int launch_gpu_search(const GpuDfa &dfa, const char *data, uint64_t len,
                      uint64_t *elapsed_ns, GpuMatch *matches,
                      int *match_count) {
    // Return error to indicate GPU search is not available
    // This will cause a fallback to CPU search
    (void)dfa;
    (void)data;
    (void)len;
    (void)matches;
    (void)match_count;
    if (elapsed_ns) {
        *elapsed_ns = 0;
    }
    return -1; // Error
}
#endif
