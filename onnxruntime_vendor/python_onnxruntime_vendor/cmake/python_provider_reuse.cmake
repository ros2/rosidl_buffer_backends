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

set(_python_provider_probe "${CMAKE_CURRENT_LIST_DIR}/probe_python.py")
function(python_provider_find_compatible module version require_cuda output_root output_variant)
  set(${output_root} "" PARENT_SCOPE)
  if(ARGC GREATER 5)
    set(${ARGV5} "" PARENT_SCOPE)
  endif()
  if(FORCE_BUILD_VENDOR_PKG OR CMAKE_CROSSCOMPILING)
    return()
  endif()
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${_python_provider_probe}"
    "${module}" "${version}" "${require_cuda}" "${CUDAToolkit_VERSION_MAJOR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
    OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 60)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/python_reuse.log" "${output}\n${error}")
  if(result EQUAL 0)
    string(REPLACE "\n" ";" fields "${output}")
    list(GET fields 0 root)
    list(GET fields 1 variant)
    list(GET fields 2 selected_version)
    if(ARGC GREATER 5)
      set(${ARGV5} "${selected_version}" PARENT_SCOPE)
    endif()
    set(${output_root} "${root}" PARENT_SCOPE)
    set(${output_variant} "${variant}" PARENT_SCOPE)
    message(STATUS "${PROJECT_NAME}: reusing ${module} ${selected_version} (${variant}) at ${root}")
  else()
    message(STATUS "${PROJECT_NAME}: no compatible installed ${module}; staging pinned wheel")
  endif()
endfunction()

macro(python_provider_reuse_hooks root)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reused_python.dsv"
    "prepend-non-duplicate;PYTHONPATH;${root}\n")
  ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/reused_python.dsv")
  file(GLOB_RECURSE runtimes "${root}/nvidia/libcudart.so.*")
  set(dirs "")
  foreach(runtime IN LISTS runtimes)
    get_filename_component(dir "${runtime}" DIRECTORY)
    list(APPEND dirs "${dir}")
  endforeach()
  list(REMOVE_DUPLICATES dirs)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reused_cuda.dsv" "")
  foreach(dir IN LISTS dirs)
    file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/reused_cuda.dsv"
      "append-non-duplicate;LD_LIBRARY_PATH;${dir}\n")
  endforeach()
  ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/reused_cuda.dsv")
endmacro()
