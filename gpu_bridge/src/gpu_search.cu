#include "gpu_search.cuh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

// Simple naive search kernel
// Each thread checks if the pattern matches starting at its index.
__global__ void search_kernel(const char *haystack, uint64_t haystack_len,
                              const char *needle, uint32_t needle_len,
                              bool dotall,
                              int *result // 0 = not found, 1 = found
) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= haystack_len || idx + needle_len > haystack_len) {
    return;
  }

  // Optimization: check first char first
  if (!dotall && haystack[idx] != needle[0] && needle[0] != '.') {
    return;
  }

  bool match = true;
  for (uint32_t i = 0; i < needle_len; ++i) {
    char h = haystack[idx + i];
    char n = needle[i];
    if (n != '.' || !dotall) {
      if (h != n) {
        match = false;
        break;
      }
    }
  }

  if (match) {
    *result = 1;
  }
}

bool launch_gpu_search(const GpuPattern &pattern, const char *file_path,
                       uint64_t file_len, uint64_t *elapsed_ns) {
  // 1. Read file into host memory
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return false;
  }

  std::vector<char> host_data(file_len);
  if (!file.read(host_data.data(), file_len)) {
    std::cerr << "Failed to read file: " << file_path << std::endl;
    return false;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  try {
    cudaEventRecord(start);

    // 2. Copy to device
    thrust::device_vector<char> d_haystack = host_data;
    thrust::device_vector<char> d_needle(pattern.pattern.begin(),
                                         pattern.pattern.end());
    thrust::device_vector<int> d_result(1, 0);

    // 3. Execute search
    int blockSize = 256;
    int numBlocks = (file_len + blockSize - 1) / blockSize;

    search_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_haystack.data()), file_len,
        thrust::raw_pointer_cast(d_needle.data()), pattern.pattern.length(),
        pattern.dotall, thrust::raw_pointer_cast(d_result.data()));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
                << std::endl;
      return false;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *elapsed_ns = static_cast<uint64_t>(milliseconds * 1000000.0f);

    int result = d_result[0];

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result == 1;

  } catch (thrust::system_error &e) {
    std::cerr << "Thrust error: " << e.what() << std::endl;
    return false;
  } catch (std::bad_alloc &e) {
    std::cerr << "Allocation error: " << e.what() << std::endl;
    return false;
  }
}
