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

if(NOT EXISTS "${PROVIDER_LIBRARY}")
  message(FATAL_ERROR "Missing libonnxruntime_providers_cuda.so")
endif()
if(NOT EXISTS "${CORE_LIBRARY}")
  message(FATAL_ERROR "Missing libonnxruntime.so")
endif()
if(NOT EXISTS "${ARCHIVE_VERSION_FILE}")
  message(FATAL_ERROR "Missing archive VERSION_NUMBER")
endif()

get_filename_component(_provider_directory "${PROVIDER_LIBRARY}" REALPATH)
get_filename_component(_provider_directory "${_provider_directory}" DIRECTORY)
get_filename_component(_core_directory "${CORE_LIBRARY}" REALPATH)
get_filename_component(_core_directory "${_core_directory}" DIRECTORY)
if(NOT _provider_directory STREQUAL _core_directory)
  message(FATAL_ERROR
    "ONNX Runtime CUDA provider must be physically colocated with the core")
endif()

file(STRINGS "${ARCHIVE_VERSION_FILE}" _archive_version LIMIT_COUNT 1)
if(NOT _archive_version STREQUAL EXPECTED_VERSION)
  message(FATAL_ERROR
    "Archive version '${_archive_version}' does not match '${EXPECTED_VERSION}'")
endif()

if(READELF AND EXISTS "${READELF}")
  execute_process(
    COMMAND "${READELF}" -d "${PROVIDER_LIBRARY}"
    RESULT_VARIABLE _readelf_result
    OUTPUT_VARIABLE _dynamic_section
    ERROR_VARIABLE _readelf_error)
  if(NOT _readelf_result EQUAL 0)
    message(FATAL_ERROR "readelf failed: ${_readelf_error}")
  endif()
  if(NOT _dynamic_section MATCHES
      "NEEDED.*(cuda|cudnn|cublas|cufft|curand|nvrtc|nvJitLink)")
    message(FATAL_ERROR "CUDA provider has no CUDA runtime dependency")
  endif()
endif()
