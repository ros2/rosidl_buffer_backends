# Copyright 2026 Open Source Robotics Foundation, Inc.
# Copyright 2026 NVIDIA Corporation
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

set(LIBTORCH_VENDOR_FALLBACK_VERSION "2.14.0")
set(LIBTORCH_VENDOR_MIN_VERSION "2.5.0")

macro(libtorch_vendor_find_cuda)
  find_package(CUDAToolkit REQUIRED)

  # Caffe2's prebuilt CUDA targets use torch::cudart.
  if(NOT TARGET torch::cudart)
    add_library(torch::cudart INTERFACE IMPORTED)
    set_property(TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES CUDA::cudart)
  endif()
endmacro()

macro(libtorch_vendor_select_sdk)
  get_filename_component(_selected_torch_root "${Torch_DIR}/../../.." REALPATH)
  if(DEFINED ENV{TORCH_INSTALL_PREFIX})
    get_filename_component(_environment_torch_root "$ENV{TORCH_INSTALL_PREFIX}" REALPATH)
    if(NOT _environment_torch_root STREQUAL _selected_torch_root)
      message(FATAL_ERROR "TORCH_INSTALL_PREFIX conflicts with the selected vendor; unset it or select that SDK when building the vendor")
    endif()
  endif()
  if(TARGET torch)
    get_target_property(_existing_torch torch IMPORTED_LOCATION_RELEASE)
    if(_existing_torch)
      get_filename_component(_existing_torch_root "${_existing_torch}/../.." REALPATH)
      if(NOT _existing_torch_root STREQUAL _selected_torch_root)
        message(FATAL_ERROR "A different Torch SDK already defined imported targets; find the selected vendor before finding Torch")
      endif()
    endif()
  endif()
  set(Caffe2_DIR "${Torch_DIR}/../Caffe2")
  set(TORCH_LIBRARY "${_selected_torch_root}/lib/libtorch.so")
  set(c10_LIBRARY "${_selected_torch_root}/lib/libc10.so")
  if(EXISTS "${_selected_torch_root}/lib/libc10_cuda.so")
    set(C10_CUDA_LIBRARY "${_selected_torch_root}/lib/libc10_cuda.so")
  endif()
endmacro()

function(libtorch_vendor_stage_wheel version variant output_root)
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "${PROJECT_NAME}: wheel staging requires a native build for the target architecture")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(root "${CMAKE_CURRENT_BINARY_DIR}/native-${version}-${variant}")
  set(wheels "${CMAKE_CURRENT_BINARY_DIR}/wheels-${version}-${variant}")
  set(marker "${root}/.complete")
  if(NOT EXISTS "${marker}")
    file(REMOVE_RECURSE "${root}" "${wheels}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip download
        --disable-pip-version-check --no-cache-dir --only-binary=:all:
        --retries 10 --timeout 600 --dest "${wheels}"
        --index-url "https://download.pytorch.org/whl/${variant}"
        --extra-index-url "https://pypi.org/simple"
        "torch==${version}+${variant}"
      COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/stage_native_wheels.py"
        "${wheels}" "${root}" "${version}+${variant}"
      COMMAND_ERROR_IS_FATAL ANY)
    file(WRITE "${marker}" "")
    file(REMOVE_RECURSE "${wheels}")
  endif()
  set(${output_root} "${root}" PARENT_SCOPE)
endfunction()

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
  set(python_argument)
  if(Python3_EXECUTABLE)
    set(python_argument "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}"
    -S "${_libtorch_probe_dir}" -B "${probe_build}"
    "-DTorch_DIR=${Torch_DIR}" "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    ${cuda_root_argument} ${python_argument} "-DREQUIRE_CUDA=${require_cuda}"
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

function(libtorch_vendor_prepare_libraries root)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR EXISTS "${root}/.elf-normalized")
    return()
  endif()
  # Rewriting the ELF layout keeps debugedit from corrupting section headers
  # placed before later sections by upstream wheel/archive processing. Preserve
  # symbols so Debian's normal dh_strip can still generate debug packages.
  file(GLOB_RECURSE libraries LIST_DIRECTORIES FALSE "${root}/*.so" "${root}/*.so.*")
  foreach(library IN LISTS libraries)
    if(IS_SYMLINK "${library}")
      continue()
    endif()
    file(READ "${library}" magic LIMIT 4 HEX)
    if(magic STREQUAL "7f454c46")
      execute_process(COMMAND "${CMAKE_OBJCOPY}" --preserve-dates "${library}"
        COMMAND_ERROR_IS_FATAL ANY)
    endif()
  endforeach()
  file(WRITE "${root}/.elf-normalized" "")
endfunction()
