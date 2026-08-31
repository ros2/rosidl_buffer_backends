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

set(ONNXRUNTIME_VERSION "1.28.0")
set(ONNXRUNTIME_ARCHIVE
  "onnxruntime-linux-x64-gpu_cuda13-${ONNXRUNTIME_VERSION}.tgz")
set(ONNXRUNTIME_URL
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ARCHIVE}")
set(ONNXRUNTIME_SHA256
  "84d28f27589090b280d4312743efd3d450cd4ac7d1e1d75e7d9076d9637bf9de")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
if(NOT _system_arch MATCHES "^(x86_64|amd64)$")
  message(FATAL_ERROR
    "${PROJECT_NAME} supports only amd64/x86_64; got '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

set(_download_path "${CMAKE_CURRENT_BINARY_DIR}/${ONNXRUNTIME_ARCHIVE}")
set(_extract_path "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_extracted")
set(_onnxruntime_root
  "${_extract_path}/onnxruntime-linux-x64-gpu_cuda13-${ONNXRUNTIME_VERSION}")

if(NOT EXISTS "${_onnxruntime_root}/lib/libonnxruntime.so")
  file(DOWNLOAD
    "${ONNXRUNTIME_URL}"
    "${_download_path}"
    EXPECTED_HASH "SHA256=${ONNXRUNTIME_SHA256}"
    SHOW_PROGRESS
    STATUS _download_status
  )
  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    list(GET _download_status 1 _download_message)
    message(FATAL_ERROR
      "${PROJECT_NAME}: ONNX Runtime download failed: ${_download_message}")
  endif()
  file(ARCHIVE_EXTRACT INPUT "${_download_path}" DESTINATION "${_extract_path}")
  file(REMOVE "${_download_path}")
endif()

foreach(_required_library
    libonnxruntime.so
    libonnxruntime_providers_shared.so
    libonnxruntime_providers_cuda.so)
  if(NOT EXISTS "${_onnxruntime_root}/lib/${_required_library}")
    message(FATAL_ERROR
      "${PROJECT_NAME}: archive is missing ${_required_library}")
  endif()
endforeach()

install(
  DIRECTORY "${_onnxruntime_root}/include/"
  DESTINATION "opt/${PROJECT_NAME}/include"
)
install(
  DIRECTORY "${_onnxruntime_root}/lib/"
  DESTINATION "opt/${PROJECT_NAME}/lib"
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING
    PATTERN "libonnxruntime.so*"
    PATTERN "libonnxruntime_providers_shared.so*"
    PATTERN "libonnxruntime_providers_cuda.so*"
)
install(
  FILES LICENSE_ONNXRUNTIME README.md
  DESTINATION "share/${PROJECT_NAME}"
)

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
  "${ament_cmake_vendor_package_DIR}/templates/vendor_package.dsv.in"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST "onnxruntime_cuda_vendor-extras.cmake.in")
