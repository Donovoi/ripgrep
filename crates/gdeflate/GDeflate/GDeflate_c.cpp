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

#include "GDeflate_c.h"
#include "GDeflate.h"

extern "C" {

size_t gdeflate_compress_bound(size_t size) {
    return GDeflate::CompressBound(size);
}

gdeflate_result gdeflate_compress(
    uint8_t* output,
    size_t* output_size,
    const uint8_t* input,
    size_t input_size,
    uint32_t level,
    uint32_t flags)
{
    if (!output || !output_size || !input) {
        return GDEFLATE_INVALID_PARAM;
    }
    
    if (level < GDEFLATE_MIN_COMPRESSION_LEVEL || level > GDEFLATE_MAX_COMPRESSION_LEVEL) {
        return GDEFLATE_INVALID_PARAM;
    }
    
    bool success = GDeflate::Compress(output, output_size, input, input_size, level, flags);
    return success ? GDEFLATE_SUCCESS : GDEFLATE_ERROR;
}

gdeflate_result gdeflate_decompress(
    uint8_t* output,
    size_t output_size,
    const uint8_t* input,
    size_t input_size,
    uint32_t num_workers)
{
    if (!output || !input) {
        return GDEFLATE_INVALID_PARAM;
    }
    
    bool success = GDeflate::Decompress(output, output_size, input, input_size, num_workers);
    return success ? GDEFLATE_SUCCESS : GDEFLATE_ERROR;
}

const char* gdeflate_version(void) {
    return "1.0.0";
}

} // extern "C"
