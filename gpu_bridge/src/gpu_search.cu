#include "gpu_search.cuh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

// Optimized kernel using shared memory
__global__ void search_kernel_shared(const char *haystack,
                                     uint64_t haystack_len, const char *needle,
                                     uint32_t needle_len, bool dotall,
                                     GpuMatch *matches, int *match_count,
                                     int max_matches) {
  extern __shared__ char s_mem[];

  // Layout: [Haystack Chunk (blockDim.x + needle_len)] [Needle (needle_len)]
  char *s_haystack = s_mem;
  char *s_needle = s_mem + blockDim.x + needle_len;

  uint32_t tid = threadIdx.x;
  uint64_t global_idx = blockIdx.x * blockDim.x + tid;

  // 1. Load Needle (cooperative)
  if (tid < needle_len) {
    s_needle[tid] = needle[tid];
  }

  // 2. Load Haystack Chunk
  if (global_idx < haystack_len) {
    s_haystack[tid] = haystack[global_idx];
  } else {
    s_haystack[tid] = 0;
  }

  // 3. Load Apron (bytes overlapping into next block)
  // We need 'needle_len' extra bytes at the end
  if (tid < needle_len) {
    uint64_t apron_idx = (blockIdx.x + 1) * blockDim.x + tid;
    if (apron_idx < haystack_len) {
      s_haystack[blockDim.x + tid] = haystack[apron_idx];
    } else {
      s_haystack[blockDim.x + tid] = 0;
    }
  }

  __syncthreads();

  if (global_idx >= haystack_len || global_idx + needle_len > haystack_len) {
    return;
  }

  // 4. Search using Shared Memory
  // Optimization: check first char first
  if (!dotall && s_haystack[tid] != s_needle[0] && s_needle[0] != '.') {
    return;
  }

  bool match = true;
  for (uint32_t i = 0; i < needle_len; ++i) {
    char h = s_haystack[tid + i];
    char n = s_needle[i];
    if (n != '.' || !dotall) {
      if (h != n) {
        match = false;
        break;
      }
    }
  }

  if (match) {
    int out_idx = atomicAdd(match_count, 1);
    if (out_idx < max_matches) {
      matches[out_idx].offset = global_idx;
    }
  }
}

int launch_gpu_search(const GpuPattern &pattern, const char *data, uint64_t len,
                      uint64_t *elapsed_ns, GpuMatch *matches,
                      int *match_count) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  try {
    cudaEventRecord(start);

    // Register host memory for faster transfer (pinned memory)
    // This allows the driver to DMA directly from the mmap-ed pages
    cudaError_t regErr = cudaHostRegister(const_cast<char *>(data), len,
                                          cudaHostRegisterDefault);
    if (regErr != cudaSuccess) {
      // If registration fails, we just proceed with normal pageable memory
      // It might fail if we hit lock limits
      // std::cerr << "Warning: cudaHostRegister failed: " <<
      // cudaGetErrorString(regErr) << std::endl;
    }

    thrust::device_vector<char> d_haystack(data, data + len);

    if (regErr == cudaSuccess) {
      cudaHostUnregister(const_cast<char *>(data));
    }

    thrust::device_vector<char> d_needle(pattern.pattern.begin(),
                                         pattern.pattern.end());

    thrust::device_vector<GpuMatch> d_matches(MAX_MATCHES);
    thrust::device_vector<int> d_match_count(1, 0);

    int blockSize = 256;
    int numBlocks = (len + blockSize - 1) / blockSize;

    // Calculate shared memory size
    // Haystack chunk (blockSize) + Apron (needle_len) + Needle (needle_len)
    // Note: Apron size should be needle_len - 1 strictly, but needle_len is
    // safe/aligned
    size_t sharedMemSize =
        (blockSize + 2 * pattern.pattern.length()) * sizeof(char);

    search_kernel_shared<<<numBlocks, blockSize, sharedMemSize>>>(
        thrust::raw_pointer_cast(d_haystack.data()), len,
        thrust::raw_pointer_cast(d_needle.data()), pattern.pattern.length(),
        pattern.dotall, thrust::raw_pointer_cast(d_matches.data()),
        thrust::raw_pointer_cast(d_match_count.data()), MAX_MATCHES);

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
