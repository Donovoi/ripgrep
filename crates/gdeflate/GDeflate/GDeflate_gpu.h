/**
 * GDeflate GPU Acceleration Header
 * 
 * NVIDIA CUDA-based GPU acceleration for large file decompression
 */

#ifndef GDEFLATE_GPU_H
#define GDEFLATE_GPU_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU device information structure
 */
typedef struct {
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_major;
    int compute_minor;
    int multiprocessor_count;
} GpuDeviceInfo;

/**
 * Check if GPU acceleration is available
 * 
 * @return true if GPU is available and suitable, false otherwise
 */
bool gpu_is_available(void);

/**
 * Get information about available GPU devices
 * 
 * @param devices Output array to store device information
 * @param max_devices Maximum number of devices to query
 * @return Number of devices found
 */
int gpu_get_device_info(GpuDeviceInfo* devices, int max_devices);

/**
 * Decompress data using GPU acceleration
 * 
 * @param input Pointer to compressed input data
 * @param input_size Size of compressed input in bytes
 * @param output Pointer to output buffer
 * @param output_size Expected size of decompressed data in bytes
 * @return 0 on success, negative error code on failure
 */
int gpu_decompress(const uint8_t* input, size_t input_size,
                  uint8_t* output, size_t output_size);

/**
 * Select optimal GPU device for decompression
 * 
 * @return Device ID, or -1 if no suitable device found
 */
int gpu_select_device(void);

#ifdef __cplusplus
}
#endif

#endif // GDEFLATE_GPU_H
