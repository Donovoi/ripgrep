#pragma once

#include <cstdint>
#include <string>

struct GpuPattern {
  std::string pattern;
  bool case_sensitive;
  bool dotall;
};

// Returns true if match found, false otherwise.
// Updates elapsed_ns with kernel execution time.
bool launch_gpu_search(const GpuPattern &pattern, const char *file_path,
                       uint64_t file_len, uint64_t *elapsed_ns);
