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

set(_onnxruntime_probe_source "${CMAKE_CURRENT_LIST_DIR}/check.cpp")
function(onnxruntime_vendor_find_compatible require_cuda output_include output_lib)
  set(${output_include} "" PARENT_SCOPE)
  set(${output_lib} "" PARENT_SCOPE)
  if(ARGC GREATER 3)
    set(${ARGV3} "" PARENT_SCOPE)
  endif()
  if(FORCE_BUILD_VENDOR_PKG OR CMAKE_CROSSCOMPILING)
    return()
  endif()
  find_path(probe_include onnxruntime_cxx_api.h
    HINTS "${onnxruntime_ROOT}" ENV onnxruntime_ROOT
    PATH_SUFFIXES include include/onnxruntime NO_CACHE)
  find_library(probe_library NAMES onnxruntime
    HINTS "${onnxruntime_ROOT}" ENV onnxruntime_ROOT
    PATH_SUFFIXES lib lib64 NO_CACHE)
  if(NOT probe_include OR NOT probe_library)
    return()
  endif()
  get_filename_component(probe_lib "${probe_library}" DIRECTORY)
  set(cuda_provider "${probe_lib}/libonnxruntime_providers_cuda.so")
  if(require_cuda AND NOT EXISTS "${cuda_provider}")
    return()
  endif()
  if(EXISTS "${cuda_provider}")
    file(STRINGS "${cuda_provider}" cuda_runtime REGEX "libcudart[.]so[.](12|13)" LIMIT_COUNT 1)
    if(NOT cuda_runtime)
      return()
    endif()
    if(require_cuda AND NOT cuda_runtime MATCHES "libcudart[.]so[.]${CUDAToolkit_VERSION_MAJOR}")
      return()
    endif()
  endif()
  try_run(run_result compile_result "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_probe"
    "${_onnxruntime_probe_source}"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${probe_include}" "-DCMAKE_CXX_STANDARD=17"
    LINK_LIBRARIES "${probe_library}"
    COMPILE_OUTPUT_VARIABLE output RUN_OUTPUT_VARIABLE run_output)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_reuse.log" "${output}\n${run_output}")
  if(compile_result AND run_result STREQUAL "0")
    string(REGEX MATCH "ONNXRUNTIME_VERSION=([0-9]+[.][0-9]+[.][0-9]+)" _version "${run_output}")
    set(selected_version "${CMAKE_MATCH_1}")
    if(NOT selected_version)
      message(STATUS "${PROJECT_NAME}: could not identify the existing ONNX Runtime release; staging pinned SDK")
      return()
    endif()
    if(ARGC GREATER 3)
      set(${ARGV3} "${selected_version}" PARENT_SCOPE)
    endif()
    set(${output_include} "${probe_include}" PARENT_SCOPE)
    set(${output_lib} "${probe_lib}" PARENT_SCOPE)
    message(STATUS "${PROJECT_NAME}: reusing ONNX Runtime ${selected_version} at ${probe_library}")
  else()
    string(STRIP "${run_output}" reason)
    if(NOT reason)
      set(reason "ONNX Runtime compatibility compilation failed")
    endif()
    message(STATUS "${PROJECT_NAME}: ${reason}; staging pinned SDK. "
      "Probe details: ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_reuse.log")
  endif()
endfunction()
