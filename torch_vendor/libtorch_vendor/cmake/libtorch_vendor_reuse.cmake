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

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/libtorch_vendor_common.cmake")
set(_libtorch_probe_dir "${CMAKE_CURRENT_LIST_DIR}/probe")

function(libtorch_vendor_find_compatible require_cuda output_dir output_variant)
  set(${output_dir} "" PARENT_SCOPE)
  set(${output_variant} "" PARENT_SCOPE)
  if(ARGC GREATER 3)
    set(${ARGV3} "" PARENT_SCOPE)
  endif()
  set(required_version "")
  if(ARGC GREATER 4)
    set(required_version "${ARGV4}")
  endif()
  if(FORCE_BUILD_VENDOR_PKG OR CMAKE_CROSSCOMPILING)
    return()
  endif()
  # Provider targets and configuration errors are isolated in the probe process.
  set(probe_build "${CMAKE_CURRENT_BINARY_DIR}/libtorch_reuse_probe")
  file(REMOVE_RECURSE "${probe_build}")
  set(cuda_root_argument)
  if(CUDAToolkit_ROOT)
    set(cuda_root_argument "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}"
    -S "${_libtorch_probe_dir}" -B "${probe_build}"
    "-DTorch_DIR=${Torch_DIR}" "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    ${cuda_root_argument} "-DREQUIRE_CUDA=${require_cuda}"
    "-DREQUIRED_TORCH_VERSION=${required_version}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libtorch_reuse.log" "${output}\n${error}")
  if(result EQUAL 0)
    file(STRINGS "${probe_build}/compatible.txt" compatible)
    list(GET compatible 0 found_dir)
    list(GET compatible 1 found_variant)
    list(GET compatible 2 found_version)
    if(ARGC GREATER 3)
      set(${ARGV3} "${found_version}" PARENT_SCOPE)
    endif()
    set(${output_dir} "${found_dir}" PARENT_SCOPE)
    set(${output_variant} "${found_variant}" PARENT_SCOPE)
    message(STATUS "${PROJECT_NAME}: reusing LibTorch ${found_version}+${found_variant} at ${found_dir}")
  else()
    if(EXISTS "${probe_build}/rejection_reason.txt")
      file(READ "${probe_build}/rejection_reason.txt" reason)
    else()
      set(reason "LibTorch package configuration failed")
    endif()
    message(STATUS "${PROJECT_NAME}: ${reason} Staging pinned distribution. "
      "Probe details: ${CMAKE_CURRENT_BINARY_DIR}/libtorch_reuse.log")
  endif()
endfunction()
