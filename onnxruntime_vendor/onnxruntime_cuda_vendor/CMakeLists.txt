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
project(onnxruntime_cuda_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)
find_package(onnxruntime_core_vendor REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(ONNXRUNTIME_CUDA_VENDOR_VERSION
  "${onnxruntime_core_vendor_VERSION}" CACHE STRING
  "ONNX Runtime CUDA provider version")
set(ONNXRUNTIME_CUDA_VENDOR_VARIANT
  "${ONNXRUNTIME_CORE_VENDOR_VARIANT}" CACHE STRING
  "Official GPU archive variant: cuda12 or cuda13")
set_property(
  CACHE ONNXRUNTIME_CUDA_VENDOR_VARIANT
  PROPERTY STRINGS cuda12 cuda13)
set(ONNXRUNTIME_CUDA_VENDOR_SHA256
  "${ONNXRUNTIME_CORE_VENDOR_SHA256}" CACHE STRING
  "SHA256 of the official ONNX Runtime GPU archive")

if(NOT ONNXRUNTIME_CUDA_VENDOR_VERSION STREQUAL
    onnxruntime_core_vendor_VERSION)
  message(FATAL_ERROR
    "${PROJECT_NAME}: provider version "
    "${ONNXRUNTIME_CUDA_VENDOR_VERSION} does not match core version "
    "${onnxruntime_core_vendor_VERSION}")
endif()
if(NOT ONNXRUNTIME_CUDA_VENDOR_VARIANT STREQUAL
    ONNXRUNTIME_CORE_VENDOR_VARIANT)
  message(FATAL_ERROR
    "${PROJECT_NAME}: provider variant "
    "${ONNXRUNTIME_CUDA_VENDOR_VARIANT} does not match core variant "
    "${ONNXRUNTIME_CORE_VENDOR_VARIANT}")
endif()
if(NOT ONNXRUNTIME_CUDA_VENDOR_SHA256 STREQUAL
    ONNXRUNTIME_CORE_VENDOR_SHA256)
  message(FATAL_ERROR
    "${PROJECT_NAME}: provider checksum does not match the core archive")
endif()

if(ONNXRUNTIME_CUDA_VENDOR_VARIANT STREQUAL "cuda12")
  set(_required_cuda_major 12)
elseif(ONNXRUNTIME_CUDA_VENDOR_VARIANT STREQUAL "cuda13")
  set(_required_cuda_major 13)
else()
  message(FATAL_ERROR
    "${PROJECT_NAME}: ONNXRUNTIME_CUDA_VENDOR_VARIANT must be "
    "cuda12 or cuda13")
endif()
if(NOT CUDAToolkit_VERSION_MAJOR EQUAL _required_cuda_major)
  message(FATAL_ERROR
    "${PROJECT_NAME}: ${ONNXRUNTIME_CUDA_VENDOR_VARIANT} requires "
    "CUDA ${_required_cuda_major}, but found ${CUDAToolkit_VERSION}")
endif()

set(_download_path
  "${CMAKE_CURRENT_BINARY_DIR}/${ONNXRUNTIME_CORE_VENDOR_ARCHIVE}")
set(_extract_path "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_extracted")
set(_onnxruntime_root
  "${_extract_path}/${ONNXRUNTIME_CORE_VENDOR_ARCHIVE_STEM}")

if(NOT EXISTS
    "${_onnxruntime_root}/lib/libonnxruntime_providers_cuda.so")
  set(_hash_args)
  if(ONNXRUNTIME_CUDA_VENDOR_SHA256)
    list(APPEND _hash_args
      EXPECTED_HASH
        "SHA256=${ONNXRUNTIME_CUDA_VENDOR_SHA256}")
  else()
    message(WARNING
      "${PROJECT_NAME}: no checksum is known for "
      "${ONNXRUNTIME_CORE_VENDOR_ARCHIVE}")
  endif()
  file(DOWNLOAD "${ONNXRUNTIME_CORE_VENDOR_URL}" "${_download_path}"
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

set(_provider_library
  "${_onnxruntime_root}/lib/libonnxruntime_providers_cuda.so")
if(NOT EXISTS "${_provider_library}")
  message(FATAL_ERROR
    "${PROJECT_NAME}: archive is missing libonnxruntime_providers_cuda.so")
endif()

install(FILES "${_provider_library}"
  DESTINATION "opt/onnxruntime_core_vendor/lib")
install(
  FILES "${_onnxruntime_root}/LICENSE"
  DESTINATION "share/${PROJECT_NAME}")

ament_index_register_resource(
  "vendor_packages" CONTENT "opt/onnxruntime_core_vendor")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(
  CONFIG_EXTRAS_POST "onnxruntime_cuda_vendor-extras.cmake.in")
