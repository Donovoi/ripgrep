#include "FloatCompat.h"

#include <stdint.h>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

namespace {

struct SubstringPredicate {
  const uint8_t *text;
  const uint8_t *pattern;
  size_t pattern_len;

  __device__ bool operator()(size_t pos) const {
    for (size_t i = 0; i < pattern_len; ++i) {
      if (text[pos + i] != pattern[i]) {
        return false;
      }
    }
    return true;
  }
};

inline int make_error(cudaError_t err) {
  if (err == cudaSuccess) {
    return 0;
  }
  return static_cast<int>(err);
}

} // namespace

extern "C" int gpu_substring_contains(const uint8_t *haystack,
                                      size_t haystack_len,
                                      const uint8_t *needle,
                                      size_t needle_len) {
  if (needle_len == 0) {
    // Empty pattern trivially matches.
    return 1;
  }
  if (haystack_len == 0 || haystack == nullptr || needle == nullptr) {
    return 0;
  }
  if (haystack_len < needle_len) {
    return 0;
  }

  try {
    thrust::device_vector<uint8_t> d_text(haystack_len);
    thrust::device_vector<uint8_t> d_pattern(needle_len);

    auto err = cudaMemcpy(thrust::raw_pointer_cast(d_text.data()), haystack,
                          haystack_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      return -make_error(err);
    }

    err = cudaMemcpy(thrust::raw_pointer_cast(d_pattern.data()), needle,
                     needle_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      return -make_error(err);
    }

    const size_t max_pos = haystack_len - needle_len + 1;
    auto begin = thrust::make_counting_iterator<size_t>(0);
    auto end = begin + max_pos;
    const SubstringPredicate predicate{
        thrust::raw_pointer_cast(d_text.data()),
        thrust::raw_pointer_cast(d_pattern.data()),
        needle_len,
    };

    bool found = thrust::any_of(thrust::device, begin, end, predicate);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      return -make_error(err);
    }

    return found ? 1 : 0;
  } catch (...) {
    return -1;
  }
}
