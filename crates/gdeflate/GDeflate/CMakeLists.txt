#
# SPDX-FileCopyrightText: Copyright (c) 2020, 2021, 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

include("../3rdparty/libdeflate.cmake")

set(SOURCES
  GDeflateCompress.cpp
  GDeflateDecompress.cpp
  GDeflate_c.cpp
)

set(HEADERS
  config.h
  TileStream.h
  Utils.h
)

set(PUBLIC_HEADERS
  GDeflate.h
  GDeflate_c.h
)

add_library(GDeflate STATIC ${SOURCES} ${HEADERS} ${PUBLIC_HEADERS})
target_include_directories(GDeflate PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(GDeflate PRIVATE cxx_std_17)
target_link_libraries(GDeflate PUBLIC libdeflate_static)

if (WIN32)
  target_link_libraries(GDeflate PRIVATE runtimeobject.lib)
endif()

# Also create a shared library version
add_library(GDeflate_shared SHARED ${SOURCES} ${HEADERS} ${PUBLIC_HEADERS})
target_include_directories(GDeflate_shared PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(GDeflate_shared PRIVATE cxx_std_17)
target_link_libraries(GDeflate_shared PUBLIC libdeflate_static)
set_target_properties(GDeflate_shared PROPERTIES 
  OUTPUT_NAME GDeflate
  VERSION 1.0.0
  SOVERSION 1
)

if (WIN32)
  target_link_libraries(GDeflate_shared PRIVATE runtimeobject.lib)
endif()

# Install targets
install(TARGETS GDeflate GDeflate_shared libdeflate_static
  EXPORT GDeflateTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
  PUBLIC_HEADER DESTINATION include/gdeflate
)

install(FILES ${PUBLIC_HEADERS} 
  DESTINATION include/gdeflate
)

# Install libdeflate header
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/libdeflate/libdeflate.h
  DESTINATION include/gdeflate
)

# Generate and install CMake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/GDeflateConfigVersion.cmake"
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion
)

install(EXPORT GDeflateTargets
  FILE GDeflateTargets.cmake
  NAMESPACE GDeflate::
  DESTINATION lib/cmake/GDeflate
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/GDeflateConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/GDeflateConfig.cmake"
  @ONLY
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/GDeflateConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/GDeflateConfigVersion.cmake"
  DESTINATION lib/cmake/GDeflate
)

# Generate and install pkg-config file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gdeflate.pc.in
  "${CMAKE_CURRENT_BINARY_DIR}/gdeflate.pc"
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gdeflate.pc"
  DESTINATION lib/pkgconfig
)
