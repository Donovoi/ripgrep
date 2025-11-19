#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GpuDfa {
  std::vector<uint32_t> table;
};

// Returns 1 if match found, 0 if not found, -1 on error.
// Updates elapsed_ns with kernel execution time.
// Maximum number of matches to return per search
const int MAX_MATCHES = 4096;

struct GpuMatch {
  uint64_t offset;
};

int launch_gpu_search(const GpuDfa &dfa, const char *data, uint64_t len,
                      uint64_t *elapsed_ns, GpuMatch *matches,
                      int *match_count);
