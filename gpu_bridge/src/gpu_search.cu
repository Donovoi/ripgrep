#include "gpu_search.cuh"
#include <chrono>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define THREADS_PER_BLOCK 256
#define CHUNK_SIZE 16384
#define LOOKBACK_SIZE 1024
#define STREAM_CHUNK_SIZE (256 * 1024 * 1024) // 256 MB

// Parallel DFA kernel with speculative lookback
__global__ void dfa_search_kernel(const uint32_t *table, const char *haystack,
                                  uint64_t haystack_len, GpuMatch *matches,
                                  int *match_count, int max_matches,
                                  uint64_t global_offset,
                                  uint64_t valid_start_idx) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t chunk_start = tid * CHUNK_SIZE;
  uint64_t chunk_end = chunk_start + CHUNK_SIZE;

  if (chunk_start >= haystack_len)
    return;
  if (chunk_end > haystack_len)
    chunk_end = haystack_len;

  // Lookback phase: Start early to converge state
  uint64_t idx = chunk_start;
  uint32_t state = 0;

  if (chunk_start > 0) {
    uint64_t lookback =
        (chunk_start > LOOKBACK_SIZE) ? LOOKBACK_SIZE : chunk_start;
    idx = chunk_start - lookback;
  }

  // Scan loop
  while (idx < chunk_end) {
    uint8_t byte = static_cast<uint8_t>(haystack[idx]);

    // Table lookup: state * 256 + byte
    size_t table_idx = (size_t)state * 256 + byte;
    uint32_t entry = table[table_idx];

    if (entry & 0x80000000) {
      // Match found!
      // Only report if the match ends within our assigned chunk.
      // AND if it is within the valid region of the stream chunk.
      if (idx >= chunk_start && idx >= valid_start_idx) {
        // SAFETY: Use atomic increment with early bounds check to prevent buffer overflow.
        // The buffer is allocated with extra space (MATCH_BUFFER_SIZE) to handle
        // in-flight atomic increments that may temporarily exceed max_matches.
        int old = atomicAdd(match_count, 1);
        // Only write to buffer if we got a valid slot
        // Note: Some matches may be lost if we exceed max_matches, but no buffer overflow
        if (old < max_matches) {
          matches[old].offset = global_offset + idx;
        }
        // If we've exceeded the limit, subsequent threads will also see old >= max_matches
        // and won't write, preventing overflow
      }
      state = entry & 0x7FFFFFFF;
    } else {
      state = entry;
    }

    idx++;
  }
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

    // Output buffers - allocate extra space to handle race condition window
    thrust::device_vector<GpuMatch> d_matches(MATCH_BUFFER_SIZE);
    thrust::device_vector<int> d_match_count(1, 0);

    // Allocate GPU buffer for streaming
    // We allocate slightly more for overlap
    size_t gpu_buf_size = STREAM_CHUNK_SIZE + LOOKBACK_SIZE;
    thrust::device_vector<char> d_haystack(gpu_buf_size);

    uint64_t total_processed = 0;

    while (total_processed < len) {
      // Determine chunk size
      uint64_t current_chunk_len =
          std::min((uint64_t)STREAM_CHUNK_SIZE, len - total_processed);

      // Determine copy range
      uint64_t copy_start = total_processed;
      uint64_t copy_len = current_chunk_len;
      uint64_t overlap_bytes = 0;

      if (total_processed > 0) {
        uint64_t lookback = std::min((uint64_t)LOOKBACK_SIZE, total_processed);
        copy_start -= lookback;
        copy_len += lookback;
        overlap_bytes = lookback;
      }

      // Copy to GPU
      cudaMemcpy(thrust::raw_pointer_cast(d_haystack.data()), data + copy_start,
                 copy_len, cudaMemcpyHostToDevice);

      // Calculate grid
      int total_threads = (copy_len + CHUNK_SIZE - 1) / CHUNK_SIZE;
      int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

      dfa_search_kernel<<<blocks, THREADS_PER_BLOCK>>>(
          thrust::raw_pointer_cast(d_table.data()),
          thrust::raw_pointer_cast(d_haystack.data()), copy_len,
          thrust::raw_pointer_cast(d_matches.data()),
          thrust::raw_pointer_cast(d_match_count.data()), MAX_MATCHES,
          copy_start,   // global_offset
          overlap_bytes // valid_start_idx
      );

      total_processed += current_chunk_len;
    }

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
