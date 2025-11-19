#include "gpu_search.cuh"
#include <chrono>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Simple DFA kernel
// We use a single thread for now to ensure correctness.
// Parallelizing DFA is complex (requires speculative execution or parallel
// prefix scan).
__global__ void dfa_search_kernel(const uint32_t *table, const char *haystack,
                                  uint64_t haystack_len, GpuMatch *matches,
                                  int *match_count, int max_matches) {
  // We only run on thread 0 of block 0
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  uint64_t idx = 0;
  uint32_t state = 0; // Start state is always 0 in our flattened table

  int local_match_count = 0;

  while (idx < haystack_len) {
    uint8_t byte = static_cast<uint8_t>(haystack[idx]);

    // Table lookup: state * 256 + byte
    // We use __ldg to hint read-only cache if possible, but standard access is
    // fine for now Cast to size_t to avoid overflow
    size_t table_idx = (size_t)state * 256 + byte;
    uint32_t entry = table[table_idx];

    // Check match bit (high bit)
    if (entry & 0x80000000) {
      if (local_match_count < max_matches) {
        matches[local_match_count].offset = idx;
        local_match_count++;
      }

      // Clear the match bit to get the next state ID
      state = entry & 0x7FFFFFFF;
    } else {
      state = entry;
    }

    idx++;
  }

  *match_count = local_match_count;
}

int launch_gpu_search(const GpuDfa &dfa, const char *data, uint64_t len,
                      uint64_t *elapsed_ns, GpuMatch *matches,
                      int *match_count) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  try {
    cudaEventRecord(start);

    // Copy DFA table to GPU
    thrust::device_vector<uint32_t> d_table = dfa.table;

    // Copy Haystack to GPU
    // Note: For very large files, we should stream chunks.
    // But for now, we load the whole file.
    thrust::device_vector<char> d_haystack(data, data + len);

    // Output buffers
    thrust::device_vector<GpuMatch> d_matches(MAX_MATCHES);
    thrust::device_vector<int> d_match_count(1, 0);

    // Launch kernel
    // 1 block, 1 thread
    dfa_search_kernel<<<1, 1>>>(thrust::raw_pointer_cast(d_table.data()),
                                thrust::raw_pointer_cast(d_haystack.data()),
                                len, thrust::raw_pointer_cast(d_matches.data()),
                                thrust::raw_pointer_cast(d_match_count.data()),
                                MAX_MATCHES);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
                << std::endl;
      return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *elapsed_ns = static_cast<uint64_t>(milliseconds * 1000000.0f);

    // Copy results back
    int count = d_match_count[0];
    *match_count = std::min(count, MAX_MATCHES);

    if (*match_count > 0) {
      thrust::copy_n(d_matches.begin(), *match_count, matches);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (*match_count > 0) ? 1 : 0;

  } catch (thrust::system_error &e) {
    std::cerr << "Thrust error: " << e.what() << std::endl;
    return -1;
  } catch (std::bad_alloc &e) {
    std::cerr << "Allocation error: " << e.what() << std::endl;
    return -1;
  }
}
