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
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(ONNXRUNTIME_CUDA_VENDOR_VERSION "1.26.0" CACHE STRING
  "ONNX Runtime version to download")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
if(NOT _system_arch MATCHES "^(x86_64|amd64)$")
  message(FATAL_ERROR
    "${PROJECT_NAME}: ONNX Runtime GPU archives support x86_64 only")
endif()
if(NOT CUDAToolkit_VERSION_MAJOR EQUAL 12)
  message(FATAL_ERROR
    "${PROJECT_NAME}: ONNX Runtime ${ONNXRUNTIME_CUDA_VENDOR_VERSION} "
    "requires CUDA 12, but found CUDA ${CUDAToolkit_VERSION}")
endif()

set(ONNXRUNTIME_CUDA_VENDOR_ARCHIVE
  "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_CUDA_VENDOR_VERSION}.tgz")
set(ONNXRUNTIME_CUDA_VENDOR_URL
  "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_CUDA_VENDOR_VERSION}/${ONNXRUNTIME_CUDA_VENDOR_ARCHIVE}")

# ONNX Runtime's CUDA 12 release is built with CUDA 12.8. NVIDIA's minor
# version compatibility covers the installed driver, but it does not add new
# entry points to an older libcudart (Ubuntu Resolute currently ships 12.4).
# Keep that ABI detail inside the exclusive runtime provider instead of making
# every conversion plugin know which CUDA minor built ONNX Runtime.
set(_cuda_runtime_root "")
if(CUDAToolkit_VERSION VERSION_LESS "12.8")
  set(_cuda_runtime_version "12.8.90")
  set(_cuda_runtime_package "nvidia-cuda-runtime-cu12")
  set(_cuda_runtime_download_dir
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_wheel")
  set(_cuda_runtime_root
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime_extracted")
  if(NOT EXISTS
      "${_cuda_runtime_root}/nvidia/cuda_runtime/lib/libcudart.so.12")
    file(REMOVE_RECURSE
      "${_cuda_runtime_download_dir}" "${_cuda_runtime_root}")
    file(MAKE_DIRECTORY "${_cuda_runtime_download_dir}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip download
        --disable-pip-version-check
        --no-cache-dir
        --no-deps
        --only-binary=:all:
        --dest "${_cuda_runtime_download_dir}"
        "${_cuda_runtime_package}==${_cuda_runtime_version}"
      COMMAND_ERROR_IS_FATAL ANY)
    file(GLOB _cuda_runtime_wheels
      "${_cuda_runtime_download_dir}/*.whl")
    list(LENGTH _cuda_runtime_wheels _cuda_runtime_wheel_count)
    if(NOT _cuda_runtime_wheel_count EQUAL 1)
      message(FATAL_ERROR
        "${PROJECT_NAME}: expected one CUDA runtime wheel, found "
        "${_cuda_runtime_wheel_count}")
    endif()
    file(REMOVE_RECURSE "${_cuda_runtime_root}")
    file(ARCHIVE_EXTRACT INPUT "${_cuda_runtime_wheels}"
      DESTINATION "${_cuda_runtime_root}")
  endif()
endif()

set(_download_path
  "${CMAKE_CURRENT_BINARY_DIR}/${ONNXRUNTIME_CUDA_VENDOR_ARCHIVE}")
set(_extract_path "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_extracted")
set(_root_glob "${_extract_path}/onnxruntime-linux-x64-*")

# Releases disagree on whether the CUDA-major suffix survives into the
# extracted directory name, so discovering the root keeps a version bump from
# silently failing the required-file check below.
file(GLOB _onnxruntime_root LIST_DIRECTORIES true "${_root_glob}")

if(NOT EXISTS "${_onnxruntime_root}/lib/libonnxruntime.so")
  file(DOWNLOAD "${ONNXRUNTIME_CUDA_VENDOR_URL}" "${_download_path}"
    SHOW_PROGRESS STATUS _download_status)
  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    list(GET _download_status 1 _download_message)
    message(FATAL_ERROR
      "${PROJECT_NAME}: ONNX Runtime download failed: ${_download_message}")
  endif()
  file(REMOVE_RECURSE "${_extract_path}")
  file(ARCHIVE_EXTRACT INPUT "${_download_path}" DESTINATION "${_extract_path}")
  file(REMOVE "${_download_path}")
  file(GLOB _onnxruntime_root LIST_DIRECTORIES true "${_root_glob}")
endif()

list(LENGTH _onnxruntime_root _root_count)
if(NOT _root_count EQUAL 1)
  message(FATAL_ERROR
    "${PROJECT_NAME}: expected one extracted archive root, found ${_root_count}")
endif()

set(_required_files
  include/onnxruntime_c_api.h
  VERSION_NUMBER
  lib/libonnxruntime.so
  lib/libonnxruntime_providers_shared.so
  lib/libonnxruntime_providers_cuda.so
  lib/cmake/onnxruntime/onnxruntimeConfig.cmake
  lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake)
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
    PATTERN "libonnxruntime_providers_cuda.so*"
    PATTERN "cmake" EXCLUDE)
install(DIRECTORY "${_onnxruntime_cmake_dir}/onnxruntime"
  DESTINATION "opt/${PROJECT_NAME}/lib/cmake")
install(FILES "${_onnxruntime_root}/VERSION_NUMBER"
  DESTINATION "opt/${PROJECT_NAME}")
install(FILES "${_onnxruntime_root}/LICENSE"
  DESTINATION "share/${PROJECT_NAME}")
if(_cuda_runtime_root)
  install(DIRECTORY "${_cuda_runtime_root}/nvidia/cuda_runtime/lib/"
    DESTINATION "opt/${PROJECT_NAME}/lib"
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING PATTERN "libcudart.so*")
  file(GLOB _cuda_runtime_licenses
    "${_cuda_runtime_root}/nvidia_cuda_runtime_cu12-*.dist-info/License*")
  install(FILES ${_cuda_runtime_licenses}
    DESTINATION "share/${PROJECT_NAME}/cuda_runtime_licenses")
endif()

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

ament_package(CONFIG_EXTRAS_POST "onnxruntime_cuda_vendor-extras.cmake.in")
