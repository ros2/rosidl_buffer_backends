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
  if(require_cuda)
    execute_process(COMMAND "${CMAKE_NM}" -D --defined-only "${cuda_provider}"
      RESULT_VARIABLE symbols_result OUTPUT_VARIABLE symbols ERROR_VARIABLE symbols_error)
    if(NOT symbols_result EQUAL 0 OR NOT symbols MATCHES "CreateEpFactories" OR
        NOT symbols MATCHES "ReleaseEpFactory")
      message(STATUS "${PROJECT_NAME}: the CUDA provider lacks the plugin factory API; staging pinned SDK")
      return()
    endif()
  endif()
  set(selected_version "")
  set(version_config "${probe_lib}/cmake/onnxruntime/onnxruntimeConfigVersion.cmake")
  if(EXISTS "${version_config}")
    set(PACKAGE_VERSION "")
    include("${version_config}")
    set(selected_version "${PACKAGE_VERSION}")
  elseif(EXISTS "${probe_lib}/../VERSION_NUMBER")
    file(READ "${probe_lib}/../VERSION_NUMBER" selected_version)
    string(STRIP "${selected_version}" selected_version)
  endif()
  if(NOT selected_version MATCHES "^[0-9]+[.][0-9]+[.][0-9]+")
    message(STATUS "${PROJECT_NAME}: the ONNX Runtime SDK has no version metadata; staging pinned SDK")
    return()
  endif()
  if(selected_version VERSION_LESS "1.23.0")
    message(STATUS "${PROJECT_NAME}: ONNX Runtime >=1.23.0 is required, found ${selected_version}; staging pinned SDK")
    return()
  endif()
  if(ARGC GREATER 3)
    set(${ARGV3} "${selected_version}" PARENT_SCOPE)
  endif()
  set(${output_include} "${probe_include}" PARENT_SCOPE)
  set(${output_lib} "${probe_lib}" PARENT_SCOPE)
  message(STATUS "${PROJECT_NAME}: reusing ONNX Runtime ${selected_version} at ${probe_library}")
endfunction()
