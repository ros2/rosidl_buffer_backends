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

function(onnxruntime_cuda_vendor_stage_wheel root)
  set(marker "${root}/.cuda-wheel-complete")
  if(EXISTS "${marker}")
    return()
  endif()
  set(wheels "${CMAKE_CURRENT_BINARY_DIR}/cuda-wheel")
  file(REMOVE_RECURSE "${wheels}")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip download
      --disable-pip-version-check --no-cache-dir --no-deps --only-binary=:all:
      --index-url "https://pypi.org/simple" --dest "${wheels}"
      "onnxruntime-gpu==${ONNXRUNTIME_CUDA_VENDOR_VERSION}"
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/stage_native_wheel.py"
      "${wheels}" "${root}" "${ONNXRUNTIME_CUDA_VENDOR_VERSION}"
      "${CUDAToolkit_VERSION_MAJOR}"
    COMMAND_ERROR_IS_FATAL ANY)
  file(WRITE "${marker}" "")
  file(REMOVE_RECURSE "${wheels}")
endfunction()
