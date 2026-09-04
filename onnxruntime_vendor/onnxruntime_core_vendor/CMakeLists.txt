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
project(onnxruntime_core_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

set(ONNXRUNTIME_CORE_VENDOR_VERSION "1.28.0" CACHE STRING
  "ONNX Runtime version to download")
set(ONNXRUNTIME_CORE_VENDOR_VARIANT "cuda12" CACHE STRING
  "Official x86_64 GPU archive variant: cuda12 or cuda13")
set_property(
  CACHE ONNXRUNTIME_CORE_VENDOR_VARIANT PROPERTY STRINGS cuda12 cuda13)
set(ONNXRUNTIME_CORE_VENDOR_SHA256 "" CACHE STRING
  "SHA256 override for the selected official ONNX Runtime archive")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
if(_system_arch MATCHES "^(x86_64|amd64)$")
  set(ONNXRUNTIME_CORE_VENDOR_ARCHIVE_KIND "x64_gpu")
  if(NOT ONNXRUNTIME_CORE_VENDOR_VARIANT MATCHES "^(cuda12|cuda13)$")
    message(FATAL_ERROR
      "${PROJECT_NAME}: ONNXRUNTIME_CORE_VENDOR_VARIANT must be cuda12 or cuda13")
  endif()
  set(ONNXRUNTIME_CORE_VENDOR_ARCHIVE_STEM
    "onnxruntime-linux-x64-gpu_${ONNXRUNTIME_CORE_VENDOR_VARIANT}-${ONNXRUNTIME_CORE_VENDOR_VERSION}")
elseif(_system_arch MATCHES "^(aarch64|arm64)$")
  set(ONNXRUNTIME_CORE_VENDOR_ARCHIVE_KIND "aarch64_cpu")
  set(ONNXRUNTIME_CORE_VENDOR_ARCHIVE_STEM
    "onnxruntime-linux-aarch64-${ONNXRUNTIME_CORE_VENDOR_VERSION}")
else()
  message(FATAL_ERROR
    "${PROJECT_NAME}: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

if(ONNXRUNTIME_CORE_VENDOR_VERSION STREQUAL "1.28.0" AND
    ONNXRUNTIME_CORE_VENDOR_SHA256 STREQUAL "")
  if(ONNXRUNTIME_CORE_VENDOR_ARCHIVE_KIND STREQUAL "aarch64_cpu")
    set(ONNXRUNTIME_CORE_VENDOR_SHA256
      "e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb")
  elseif(ONNXRUNTIME_CORE_VENDOR_VARIANT STREQUAL "cuda12")
    set(ONNXRUNTIME_CORE_VENDOR_SHA256
      "ea6bd2b65d7dfabbeb92c4af5dd8f12e5aed8601e544ad378d2f872275438b1a")
  else()
    set(ONNXRUNTIME_CORE_VENDOR_SHA256
      "84d28f27589090b280d4312743efd3d450cd4ac7d1e1d75e7d9076d9637bf9de")
  endif()
endif()

set(ONNXRUNTIME_CORE_VENDOR_ARCHIVE
  "${ONNXRUNTIME_CORE_VENDOR_ARCHIVE_STEM}.tgz")
set(ONNXRUNTIME_CORE_VENDOR_URL
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_CORE_VENDOR_VERSION}/${ONNXRUNTIME_CORE_VENDOR_ARCHIVE}")

set(_download_path
  "${CMAKE_CURRENT_BINARY_DIR}/${ONNXRUNTIME_CORE_VENDOR_ARCHIVE}")
set(_extract_path "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_extracted")
set(_onnxruntime_root
  "${_extract_path}/${ONNXRUNTIME_CORE_VENDOR_ARCHIVE_STEM}")

if(NOT EXISTS "${_onnxruntime_root}/lib/libonnxruntime.so")
  set(_hash_args)
  if(ONNXRUNTIME_CORE_VENDOR_SHA256)
    list(APPEND _hash_args
      EXPECTED_HASH "SHA256=${ONNXRUNTIME_CORE_VENDOR_SHA256}")
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

set(_required_files
  include/onnxruntime_c_api.h
  VERSION_NUMBER
  lib/libonnxruntime.so
  lib/cmake/onnxruntime/onnxruntimeConfig.cmake
  lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake
  lib/libonnxruntime_providers_shared.so)
if(ONNXRUNTIME_CORE_VENDOR_ARCHIVE_KIND STREQUAL "x64_gpu")
  list(APPEND _required_files lib/libonnxruntime_providers_cuda.so)
endif()
foreach(_required_file IN LISTS _required_files)
  if(NOT EXISTS "${_onnxruntime_root}/${_required_file}")
    message(FATAL_ERROR
      "${PROJECT_NAME}: archive is missing ${_required_file}")
  endif()
endforeach()

set(_onnxruntime_cmake_dir
  "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_cmake")
file(REMOVE_RECURSE "${_onnxruntime_cmake_dir}")
file(COPY "${_onnxruntime_root}/lib/cmake/onnxruntime"
  DESTINATION "${_onnxruntime_cmake_dir}")
file(GLOB _onnxruntime_cmake_files
  "${_onnxruntime_cmake_dir}/onnxruntime/*.cmake")
foreach(_cmake_file IN LISTS _onnxruntime_cmake_files)
  file(READ "${_cmake_file}" _cmake_contents)
  string(REPLACE
    "\${_IMPORT_PREFIX}/lib64/" "\${_IMPORT_PREFIX}/lib/"
    _cmake_contents "${_cmake_contents}")
  file(WRITE "${_cmake_file}" "${_cmake_contents}")
endforeach()

install(DIRECTORY "${_onnxruntime_root}/include/"
  DESTINATION "opt/${PROJECT_NAME}/include")
install(DIRECTORY "${_onnxruntime_root}/lib/"
  DESTINATION "opt/${PROJECT_NAME}/lib"
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "libonnxruntime.so*"
    PATTERN "libonnxruntime_providers_shared.so*"
    PATTERN "cmake" EXCLUDE)
install(DIRECTORY "${_onnxruntime_cmake_dir}/onnxruntime"
  DESTINATION "opt/${PROJECT_NAME}/lib/cmake")
install(FILES "${_onnxruntime_root}/VERSION_NUMBER"
  DESTINATION "opt/${PROJECT_NAME}")
install(
  FILES "${_onnxruntime_root}/LICENSE"
  DESTINATION "share/${PROJECT_NAME}")

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

ament_package(CONFIG_EXTRAS_POST "onnxruntime_core_vendor-extras.cmake.in")
