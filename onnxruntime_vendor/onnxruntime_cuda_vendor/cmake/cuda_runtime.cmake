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

set(_cuda_runtime_root "")
if(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION VERSION_LESS "12.8")
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

if(_cuda_runtime_root)
  install(DIRECTORY "${_cuda_runtime_root}/nvidia/cuda_runtime/lib/"
    DESTINATION "opt/${PROJECT_NAME}/lib"
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING PATTERN "libcudart.so*")
  file(GLOB _cuda_runtime_licenses
    "${_cuda_runtime_root}/nvidia_cuda_runtime_cu12-*.dist-info/License*")
  install(FILES ${_cuda_runtime_licenses}
    DESTINATION "share/${PROJECT_NAME}/cuda_runtime_licenses")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime.dsv"
    "prepend-non-duplicate;LD_LIBRARY_PATH;opt/${PROJECT_NAME}/lib\n")
  ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/cuda_runtime.dsv")
endif()
