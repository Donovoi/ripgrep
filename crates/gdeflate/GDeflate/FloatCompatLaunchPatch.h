// This header intentionally has no include guards: it may be included multiple
// times to reapply the __cudaLaunch override after CUDA's internal headers
// redefine the macro.

// NVCC's host pass may define __CUDA_ARCH__ while invoking the host compiler
// (e.g. for the generated *.cudafe1.cpp/*.stub.c files).  In that scenario the
// original guard ended up suppressing our compatibility shim, which meant the
// stub compilations still saw CUDA's two-argument __cudaLaunch macro.  Treat any
// translation unit that isn't compiled by NVCC itself (i.e. __NVCC__ undefined)
// as "host" so the variadic shim remains active even if __CUDA_ARCH__ is set.
#if !defined(__CUDA_ARCH__) || !defined(__NVCC__)

#ifndef __RG_NVCC_LAUNCH_HELPERS_DEFINED
#define __RG_NVCC_LAUNCH_HELPERS_DEFINED 1

#  ifndef __NV_ATTR_UNUSED_FOR_LAUNCH
#    if defined(__GNUC__)
#      define __NV_ATTR_UNUSED_FOR_LAUNCH __attribute__((unused))
#    else
#      define __NV_ATTR_UNUSED_FOR_LAUNCH
#    endif
#  endif

#  ifdef __NV_LEGACY_LAUNCH
#    define __rg_cuda_launch_two_args(__rg_fun, __rg_tile_kernel) \
	{ volatile static char *__f __NV_ATTR_UNUSED_FOR_LAUNCH; __f = (__rg_fun); \
	  dim3 __gridDim, __blockDim; \
	  size_t __sharedMem; \
	  cudaStream_t __stream; \
	  if (__cudaPopCallConfiguration(&__gridDim, &__blockDim, &__sharedMem, &__stream) != cudaSuccess) \
	    return; \
	  if (__rg_tile_kernel) \
	    __blockDim.x = __blockDim.y = __blockDim.z = 1; \
	  if (__args_idx == 0) { \
	    (void)cudaLaunchKernel(__rg_fun, __gridDim, __blockDim, &__args_arr[__args_idx], __sharedMem, __stream); \
	  } else { \
	    (void)cudaLaunchKernel(__rg_fun, __gridDim, __blockDim, &__args_arr[0], __sharedMem, __stream); \
	  } }
#  else
#    define __rg_cuda_launch_two_args(__rg_fun, __rg_tile_kernel) \
	{ volatile static char *__f __NV_ATTR_UNUSED_FOR_LAUNCH; __f = (__rg_fun); \
	  static cudaKernel_t __handle = 0; \
	  volatile static bool __tmp __NV_ATTR_UNUSED_FOR_LAUNCH = (__cudaGetKernel(&__handle, (const void *)(__rg_fun)) == cudaSuccess); \
	  dim3 __gridDim, __blockDim; \
	  size_t __sharedMem; \
	  cudaStream_t __stream; \
	  if (__cudaPopCallConfiguration(&__gridDim, &__blockDim, &__sharedMem, &__stream) != cudaSuccess) \
	    return; \
	  if (__rg_tile_kernel) \
	    __blockDim.x = __blockDim.y = __blockDim.z = 1; \
	  if (__args_idx == 0) { \
	    (void)__cudaLaunchKernel_helper(__handle, __gridDim, __blockDim, &__args_arr[__args_idx], __sharedMem, __stream); \
	  } else { \
	    (void)__cudaLaunchKernel_helper(__handle, __gridDim, __blockDim, &__args_arr[0], __sharedMem, __stream); \
	  } }
#  endif

#  define __rg_cuda_launch_one_arg(__rg_fun) \
		__rg_cuda_launch_two_args(__rg_fun, 0)

#endif  // __RG_NVCC_LAUNCH_HELPERS_DEFINED

#  ifdef __rg_cuda_launch_dispatch
#    undef __rg_cuda_launch_dispatch
#  endif
#  define __rg_cuda_launch_dispatch(_1, _2, _3, ...) _3

#  ifdef __cudaLaunch
#    undef __cudaLaunch
#  endif
#  ifdef __cplusplus
#    pragma message("FloatCompatLaunchPatch: redefining __cudaLaunch as variadic macro")
#  endif
#  define __cudaLaunch(...) \
		__rg_cuda_launch_dispatch(__VA_ARGS__, __rg_cuda_launch_two_args, __rg_cuda_launch_one_arg)(__VA_ARGS__)

#endif  // !__CUDA_ARCH__
