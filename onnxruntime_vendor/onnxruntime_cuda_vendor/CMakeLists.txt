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
find_package(CUDAToolkit QUIET)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(ONNXRUNTIME_CUDA_VENDOR_VERSION "1.29.0")

option(FORCE_BUILD_VENDOR_PKG "Stage the pinned SDK instead of reusing ONNX Runtime" OFF)
find_package(onnxruntime_core_vendor REQUIRED)
include("${onnxruntime_core_vendor_DIR}/onnxruntime_vendor_reuse.cmake")
onnxruntime_vendor_find_compatible(TRUE
  ONNXRUNTIME_REUSED_INCLUDE_DIR ONNXRUNTIME_REUSED_LIBRARY_DIR
  ONNXRUNTIME_CUDA_VENDOR_SELECTED_VERSION)
if(ONNXRUNTIME_REUSED_LIBRARY_DIR AND
    NOT ONNXRUNTIME_CUDA_VENDOR_SELECTED_VERSION STREQUAL ONNXRUNTIME_CORE_VENDOR_SELECTED_VERSION)
  message(STATUS "${PROJECT_NAME}: rejected CUDA ONNX Runtime ${ONNXRUNTIME_CUDA_VENDOR_SELECTED_VERSION}; "
    "the CPU/core provider selected ${ONNXRUNTIME_CORE_VENDOR_SELECTED_VERSION}. Staging pinned fallback.")
  set(ONNXRUNTIME_REUSED_LIBRARY_DIR "")
  set(ONNXRUNTIME_REUSED_INCLUDE_DIR "")
endif()
if(ONNXRUNTIME_REUSED_LIBRARY_DIR)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reused_onnxruntime.dsv"
    "prepend-non-duplicate;LD_LIBRARY_PATH;${ONNXRUNTIME_REUSED_LIBRARY_DIR}\n")
  ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/reused_onnxruntime.dsv")
else()
  if(NOT CUDAToolkit_FOUND OR CUDAToolkit_VERSION VERSION_LESS "13.1" OR
      NOT CUDAToolkit_VERSION VERSION_LESS "14")
    message(FATAL_ERROR "${PROJECT_NAME}: the pinned ONNX Runtime 1.29.0 CUDA fallback requires "
      "CUDA >=13.1,<14; selected toolkit is '${CUDAToolkit_VERSION}'. Install CUDA 13.1 through "
      "rosdep/APT and select it with -DCUDAToolkit_ROOT=/usr/local/cuda-13.1.")
  endif()
  set(ONNXRUNTIME_CUDA_VENDOR_SELECTED_VERSION "${ONNXRUNTIME_CUDA_VENDOR_VERSION}")
  if(NOT ONNXRUNTIME_CORE_VENDOR_SELECTED_VERSION STREQUAL ONNXRUNTIME_CUDA_VENDOR_SELECTED_VERSION)
    message(FATAL_ERROR "${PROJECT_NAME}: no compatible CUDA provider for the selected CPU/core ONNX Runtime "
      "${ONNXRUNTIME_CORE_VENDOR_SELECTED_VERSION}; the pinned CUDA fallback is ${ONNXRUNTIME_CUDA_VENDOR_SELECTED_VERSION}. "
      "Select a matching CUDA SDK with onnxruntime_ROOT, or rebuild all ONNX vendors and conversions "
      "with -DFORCE_BUILD_VENDOR_PKG=ON.")
  endif()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
  if(_system_arch MATCHES "^(x86_64|amd64)$")
    set(_archive_stem "onnxruntime-linux-x64-gpu_cuda${CUDAToolkit_VERSION_MAJOR}-${ONNXRUNTIME_CUDA_VENDOR_VERSION}")
    set(_archive_root_pattern "onnxruntime-linux-x64-*")
  elseif(_system_arch MATCHES "^(aarch64|arm64)$")
    if(CMAKE_CROSSCOMPILING)
      message(FATAL_ERROR "${PROJECT_NAME}: wheel staging requires a native build for the target architecture")
    endif()
    set(_archive_stem "onnxruntime-linux-aarch64-${ONNXRUNTIME_CUDA_VENDOR_VERSION}")
    set(_archive_root_pattern "${_archive_stem}")
  else()
    message(FATAL_ERROR "${PROJECT_NAME}: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'")
  endif()
  set(ONNXRUNTIME_CUDA_VENDOR_ARCHIVE "${_archive_stem}.tgz")
  set(ONNXRUNTIME_CUDA_VENDOR_URL
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_CUDA_VENDOR_VERSION}/${ONNXRUNTIME_CUDA_VENDOR_ARCHIVE}")

  set(_download_path
    "${CMAKE_CURRENT_BINARY_DIR}/${ONNXRUNTIME_CUDA_VENDOR_ARCHIVE}")
  set(_extract_path
    "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-${ONNXRUNTIME_CUDA_VENDOR_VERSION}-cuda${CUDAToolkit_VERSION_MAJOR}")
  set(_root_glob "${_extract_path}/${_archive_root_pattern}")

  # The archive's top-level directory can differ from its filename.
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

  file(READ "${_onnxruntime_root}/VERSION_NUMBER" _archive_version)
  string(STRIP "${_archive_version}" _archive_version)
  if(NOT _archive_version STREQUAL ONNXRUNTIME_CUDA_VENDOR_VERSION)
    message(FATAL_ERROR
      "${PROJECT_NAME}: expected ${ONNXRUNTIME_CUDA_VENDOR_VERSION}, "
      "but the archive contains ${_archive_version}")
  endif()

  if(_system_arch MATCHES "^(aarch64|arm64)$")
    include(cmake/onnxruntime_cuda_vendor_wheel.cmake)
    onnxruntime_cuda_vendor_stage_wheel("${_onnxruntime_root}")
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

  include(cmake/cudnn.cmake)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  add_test(
    NAME onnxruntime_cuda_vendor_selection
    COMMAND "${CMAKE_COMMAND}"
      -S "${CMAKE_CURRENT_SOURCE_DIR}/test/consumer"
      -B "${CMAKE_CURRENT_BINARY_DIR}/selection_test"
      "-DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX};${CMAKE_PREFIX_PATH}"
  )
endif()

ament_package(CONFIG_EXTRAS_POST "onnxruntime_cuda_vendor-extras.cmake.in")
