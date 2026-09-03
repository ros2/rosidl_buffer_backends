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

foreach(_file
    VERSION_NUMBER
    include/onnxruntime_c_api.h
    lib/libonnxruntime.so
    lib/libonnxruntime_providers_shared.so)
  if(NOT EXISTS "${ONNXRUNTIME_ROOT}/${_file}")
    message(FATAL_ERROR "Missing core archive file: ${_file}")
  endif()
endforeach()

foreach(_file
    onnxruntimeConfig.cmake
    onnxruntimeConfigVersion.cmake
    onnxruntimeTargets.cmake
    onnxruntimeTargets-release.cmake)
  if(NOT EXISTS "${ONNXRUNTIME_CMAKE_DIR}/${_file}")
    message(FATAL_ERROR "Missing staged CMake file: ${_file}")
  endif()
endforeach()
file(READ
  "${ONNXRUNTIME_CMAKE_DIR}/onnxruntimeTargets-release.cmake"
  _targets_release)
if(_targets_release MATCHES "\\$\\{_IMPORT_PREFIX\\}/lib64/")
  message(FATAL_ERROR "Staged ONNX Runtime CMake target still references lib64")
endif()
if(NOT _targets_release MATCHES "\\$\\{_IMPORT_PREFIX\\}/lib/")
  message(FATAL_ERROR "Staged ONNX Runtime CMake target does not reference lib")
endif()

file(STRINGS "${ONNXRUNTIME_ROOT}/VERSION_NUMBER" _archive_version LIMIT_COUNT 1)
if(NOT _archive_version STREQUAL EXPECTED_VERSION)
  message(FATAL_ERROR
    "Archive version '${_archive_version}' does not match '${EXPECTED_VERSION}'")
endif()

file(GLOB _cuda_provider
  "${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so*")
if(ARCHIVE_KIND STREQUAL "x64_gpu")
  if(NOT _cuda_provider)
    message(FATAL_ERROR "Selected GPU archive has no CUDA provider")
  endif()
elseif(ARCHIVE_KIND STREQUAL "aarch64_cpu")
  if(_cuda_provider)
    message(FATAL_ERROR "Selected arm64 CPU archive contains a CUDA provider")
  endif()
else()
  message(FATAL_ERROR "Unsupported archive kind '${ARCHIVE_KIND}'")
endif()

if(READELF AND EXISTS "${READELF}")
  foreach(_library
      lib/libonnxruntime.so
      lib/libonnxruntime_providers_shared.so)
    execute_process(
      COMMAND "${READELF}" -d "${ONNXRUNTIME_ROOT}/${_library}"
      RESULT_VARIABLE _readelf_result
      OUTPUT_VARIABLE _dynamic_section
      ERROR_VARIABLE _readelf_error)
    if(NOT _readelf_result EQUAL 0)
      message(FATAL_ERROR "readelf failed for ${_library}: ${_readelf_error}")
    endif()
    if(_dynamic_section MATCHES
        "NEEDED.*(cuda|cudnn|cublas|cufft|curand|nvrtc|nvJitLink)")
      message(FATAL_ERROR "${_library} has a CUDA runtime dependency")
    endif()
  endforeach()
endif()
