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
