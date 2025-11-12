/*
 * SPDX-FileCopyrightText: Copyright (c) 2020, 2021, 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version information */
#define GDEFLATE_VERSION_MAJOR 1
#define GDEFLATE_VERSION_MINOR 0
#define GDEFLATE_VERSION_PATCH 0

/* Compression levels */
#define GDEFLATE_MIN_COMPRESSION_LEVEL 1
#define GDEFLATE_MAX_COMPRESSION_LEVEL 12

/* Flags */
#define GDEFLATE_COMPRESS_SINGLE_THREAD 0x200

/* Return codes */
typedef enum {
    GDEFLATE_SUCCESS = 0,
    GDEFLATE_ERROR = -1,
    GDEFLATE_INVALID_PARAM = -2,
    GDEFLATE_BUFFER_TOO_SMALL = -3
} gdeflate_result;

/**
 * Calculate the maximum size of compressed data given input size
 * @param size Input data size in bytes
 * @return Maximum possible compressed size in bytes
 */
size_t gdeflate_compress_bound(size_t size);

/**
 * Compress data using GDeflate format
 * @param output Buffer to receive compressed data
 * @param output_size Pointer to size of output buffer. On success, updated with actual compressed size
 * @param input Input data to compress
 * @param input_size Size of input data in bytes
 * @param level Compression level (1-12, where 1 is fastest and 12 is best compression)
 * @param flags Compression flags (see GDEFLATE_COMPRESS_* constants)
 * @return GDEFLATE_SUCCESS on success, error code otherwise
 */
gdeflate_result gdeflate_compress(
    uint8_t* output,
    size_t* output_size,
    const uint8_t* input,
    size_t input_size,
    uint32_t level,
    uint32_t flags);

/**
 * Decompress data from GDeflate format
 * @param output Buffer to receive decompressed data
 * @param output_size Size of output buffer in bytes
 * @param input Compressed input data
 * @param input_size Size of compressed input data in bytes
 * @param num_workers Number of worker threads to use (0 for default)
 * @return GDEFLATE_SUCCESS on success, error code otherwise
 */
gdeflate_result gdeflate_decompress(
    uint8_t* output,
    size_t output_size,
    const uint8_t* input,
    size_t input_size,
    uint32_t num_workers);

/**
 * Get version string
 * @return Version string (e.g., "1.0.0")
 */
const char* gdeflate_version(void);

#ifdef __cplusplus
}
#endif
