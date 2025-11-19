#pragma once

#include <cstdint>
#include <string>

struct GpuPattern {
  std::string pattern;
  bool case_sensitive;
  bool dotall;
};

// Returns 1 if match found, 0 if not found, -1 on error.
// Updates elapsed_ns with kernel execution time.
int launch_gpu_search(const GpuPattern &pattern, const char *data, uint64_t len,
                      uint64_t *elapsed_ns);
