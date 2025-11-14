#define __rg_cudaLaunch_original(fun, tile) do { (void)(tile); } while (0)

#define __rg_cuda_launch_two_args(__rg_fun, __rg_tile_kernel) \
                    __rg_cudaLaunch_original(__rg_fun, __rg_tile_kernel)
#define __rg_cuda_launch_one_arg(__rg_fun) \
                    __rg_cuda_launch_two_args(__rg_fun, 0)
#define __rg_cuda_launch_dispatch(_1, _2, _3, ...) _3
#define __cudaLaunch(...) \
                    __rg_cuda_launch_dispatch(__VA_ARGS__, __rg_cuda_launch_two_args, __rg_cuda_launch_one_arg)(__VA_ARGS__)

void foo() {
    __cudaLaunch(((char*)0));
}
