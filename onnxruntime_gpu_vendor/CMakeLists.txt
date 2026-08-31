# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(onnxruntime_gpu_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

set(ONNXRUNTIME_GPU_VENDOR_VERSION "1.28.0" CACHE STRING
  "ONNX Runtime version to download")
set(ONNXRUNTIME_GPU_VENDOR_VARIANT "" CACHE STRING
  "ONNX Runtime variant: cpu, cuda12, or cuda13 (auto-detected when empty)")
set_property(CACHE ONNXRUNTIME_GPU_VENDOR_VARIANT PROPERTY STRINGS cpu cuda12 cuda13)
set(ONNXRUNTIME_GPU_VENDOR_SHA256 "" CACHE STRING
  "SHA256 override for an ONNX Runtime archive")

if(ONNXRUNTIME_GPU_VENDOR_VARIANT STREQUAL "")
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    set(ONNXRUNTIME_GPU_VENDOR_VARIANT "cpu")
  elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12)
    set(ONNXRUNTIME_GPU_VENDOR_VARIANT "cuda12")
  elseif(CUDAToolkit_VERSION_MAJOR EQUAL 13)
    set(ONNXRUNTIME_GPU_VENDOR_VARIANT "cuda13")
  else()
    message(FATAL_ERROR
      "${PROJECT_NAME}: CUDA ${CUDAToolkit_VERSION} is unsupported; "
      "select cpu, cuda12, or cuda13 with ONNXRUNTIME_GPU_VENDOR_VARIANT")
  endif()
endif()

if(NOT ONNXRUNTIME_GPU_VENDOR_VARIANT MATCHES "^(cpu|cuda12|cuda13)$")
  message(FATAL_ERROR
    "${PROJECT_NAME}: ONNXRUNTIME_GPU_VENDOR_VARIANT must be cpu, cuda12, or cuda13")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
if(_system_arch MATCHES "^(x86_64|amd64)$")
  set(_archive_arch "x64")
elseif(_system_arch MATCHES "^(aarch64|arm64)$")
  set(_archive_arch "aarch64")
else()
  message(FATAL_ERROR
    "${PROJECT_NAME}: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

if(NOT ONNXRUNTIME_GPU_VENDOR_VARIANT STREQUAL "cpu" AND
    NOT _archive_arch STREQUAL "x64")
  message(FATAL_ERROR
    "${PROJECT_NAME}: official GPU archives are available only for x86_64")
endif()

if(ONNXRUNTIME_GPU_VENDOR_VARIANT STREQUAL "cpu")
  set(_archive_stem "onnxruntime-linux-${_archive_arch}-${ONNXRUNTIME_GPU_VENDOR_VERSION}")
  set(ONNXRUNTIME_GPU_VENDOR_HAS_CUDA FALSE)
else()
  set(_archive_stem
    "onnxruntime-linux-x64-gpu_${ONNXRUNTIME_GPU_VENDOR_VARIANT}-${ONNXRUNTIME_GPU_VENDOR_VERSION}")
  set(ONNXRUNTIME_GPU_VENDOR_HAS_CUDA TRUE)
endif()

if(ONNXRUNTIME_GPU_VENDOR_VERSION STREQUAL "1.28.0" AND
    ONNXRUNTIME_GPU_VENDOR_SHA256 STREQUAL "")
  if(_archive_arch STREQUAL "aarch64")
    set(ONNXRUNTIME_GPU_VENDOR_SHA256
      "e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb")
  elseif(ONNXRUNTIME_GPU_VENDOR_VARIANT STREQUAL "cpu")
    set(ONNXRUNTIME_GPU_VENDOR_SHA256
      "a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407")
  elseif(ONNXRUNTIME_GPU_VENDOR_VARIANT STREQUAL "cuda12")
    set(ONNXRUNTIME_GPU_VENDOR_SHA256
      "ea6bd2b65d7dfabbeb92c4af5dd8f12e5aed8601e544ad378d2f872275438b1a")
  else()
    set(ONNXRUNTIME_GPU_VENDOR_SHA256
      "84d28f27589090b280d4312743efd3d450cd4ac7d1e1d75e7d9076d9637bf9de")
  endif()
endif()

set(_archive "${_archive_stem}.tgz")
set(_url
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_GPU_VENDOR_VERSION}/${_archive}")
set(_download_path "${CMAKE_CURRENT_BINARY_DIR}/${_archive}")
set(_extract_path "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_extracted")
set(_onnxruntime_root "${_extract_path}/${_archive_stem}")

if(NOT EXISTS "${_onnxruntime_root}/lib/libonnxruntime.so")
  set(_hash_args)
  if(ONNXRUNTIME_GPU_VENDOR_SHA256)
    list(APPEND _hash_args
      EXPECTED_HASH "SHA256=${ONNXRUNTIME_GPU_VENDOR_SHA256}")
  else()
    message(WARNING "${PROJECT_NAME}: no checksum is known for ${_archive}")
  endif()
  file(DOWNLOAD "${_url}" "${_download_path}"
    ${_hash_args} SHOW_PROGRESS STATUS _download_status)
  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    list(GET _download_status 1 _download_message)
    message(FATAL_ERROR
      "${PROJECT_NAME}: ONNX Runtime download failed: ${_download_message}")
  endif()
  file(ARCHIVE_EXTRACT INPUT "${_download_path}" DESTINATION "${_extract_path}")
  file(REMOVE "${_download_path}")
endif()

if(NOT EXISTS "${_onnxruntime_root}/lib/libonnxruntime.so")
  message(FATAL_ERROR "${PROJECT_NAME}: archive is missing libonnxruntime.so")
endif()

install(DIRECTORY "${_onnxruntime_root}/include/"
  DESTINATION "opt/${PROJECT_NAME}/include")
install(DIRECTORY "${_onnxruntime_root}/lib/"
  DESTINATION "opt/${PROJECT_NAME}/lib"
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "libonnxruntime.so*"
    PATTERN "libonnxruntime_providers_shared.so*"
    PATTERN "libonnxruntime_providers_cuda.so*")
install(FILES LICENSE_ONNXRUNTIME README.md DESTINATION "share/${PROJECT_NAME}")

ament_index_register_resource("vendor_packages" CONTENT "opt/${PROJECT_NAME}")
if(APPLE)
  set(_LIBDIR_VAR_NAME "DYLD_LIBRARY_PATH")
elseif(WIN32)
  set(_LIBDIR_VAR_NAME "PATH")
else()
  set(_LIBDIR_VAR_NAME "LD_LIBRARY_PATH")
endif()
ament_environment_hooks(
  "${ament_cmake_vendor_package_DIR}/templates/vendor_package.sh.in"
  "${ament_cmake_vendor_package_DIR}/templates/vendor_package.dsv.in")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST "onnxruntime_gpu_vendor-extras.cmake.in")
