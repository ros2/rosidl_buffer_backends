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

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/python3_torch_cuda_vendor_policy.cmake")
if(DEFINED TEST_VARIANT AND DEFINED TEST_CUDA_VERSION)
  python3_torch_cuda_vendor_validate_variant_for_cuda("${TEST_VARIANT}" "${TEST_CUDA_VERSION}")
  return()
endif()
if(DEFINED TEST_VARIANT)
  python3_torch_cuda_vendor_validate_variant("${TEST_VARIANT}")
  return()
endif()
if(DEFINED TEST_CUDA_VERSION)
  python3_torch_cuda_vendor_variant_for_cuda("${TEST_CUDA_VERSION}" variant)
  return()
endif()
foreach(variant IN ITEMS cu126 cu130 cu132)
  python3_torch_cuda_vendor_validate_variant("${variant}")
endforeach()
foreach(pair IN ITEMS "12.6,cu126" "12.6.77,cu126" "12.8,cu126"
    "13.0,cu130" "13.1,cu130" "13.2,cu132" "13.3,cu132")
  string(REPLACE "," ";" fields "${pair}")
  list(GET fields 0 version)
  list(GET fields 1 expected)
  python3_torch_cuda_vendor_variant_for_cuda("${version}" variant)
  if(NOT variant STREQUAL expected)
    message(FATAL_ERROR "CUDA ${version} must use ${expected}")
  endif()
endforeach()
foreach(argument IN ITEMS "-DTEST_VARIANT=cu118" "-DTEST_VARIANT=cu128"
    "-DTEST_CUDA_VERSION=11.8" "-DTEST_CUDA_VERSION=12.4"
    "-DTEST_CUDA_VERSION=12.5.99" "-DTEST_CUDA_VERSION=14.0")
  execute_process(COMMAND "${CMAKE_COMMAND}" "${argument}" -P
    "${CMAKE_CURRENT_LIST_FILE}" RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
  if(result EQUAL 0)
    message(FATAL_ERROR "${argument} must be rejected")
  endif()
endforeach()
foreach(pair IN ITEMS "cu126,12.6" "cu126,12.8" "cu130,13.1" "cu130,13.2" "cu132,13.2")
  string(REPLACE "," ";" fields "${pair}")
  list(GET fields 0 variant)
  list(GET fields 1 version)
  python3_torch_cuda_vendor_validate_variant_for_cuda("${variant}" "${version}")
endforeach()
foreach(pair IN ITEMS
    "cu126,12.4,outside the supported range"
    "cu132,13.1,Torch CUDA build is newer"
    "cu130,12.6,CUDA major versions differ"
    "cu126,13.1,CUDA major versions differ")
  string(REPLACE "," ";" fields "${pair}")
  list(GET fields 0 variant)
  list(GET fields 1 version)
  list(GET fields 2 expected_reason)
  execute_process(COMMAND "${CMAKE_COMMAND}"
    "-DTEST_VARIANT=${variant}" "-DTEST_CUDA_VERSION=${version}" -P
    "${CMAKE_CURRENT_LIST_FILE}" RESULT_VARIABLE result OUTPUT_QUIET ERROR_VARIABLE error)
  if(result EQUAL 0)
    message(FATAL_ERROR "${pair} must be rejected")
  endif()
  string(REGEX REPLACE "[ \t\r\n]+" " " error "${error}")
  foreach(expected IN ITEMS "Rejected Torch ${variant}" "selected CUDA toolkit is ${version}"
      "${expected_reason}" "CUDAToolkit_ROOT")
    string(FIND "${error}" "${expected}" index)
    if(index EQUAL -1)
      message(FATAL_ERROR "Rejection must explain '${expected}': ${error}")
    endif()
  endforeach()
endforeach()

foreach(pair IN ITEMS "cu132,13.1" "cu126,12.4" "cu130,12.8" "cpu,13.1")
  string(REPLACE "," ";" fields "${pair}")
  list(GET fields 0 variant)
  list(GET fields 1 version)
  python3_torch_cuda_vendor_cuda_compatible("${variant}" "${version}" compatible)
  if(compatible)
    message(FATAL_ERROR "Incompatible installed CUDA build ${pair} must not be reused")
  endif()
endforeach()
python3_torch_cuda_vendor_cuda_compatible("cu130" "13.1.115" compatible reason)
if(NOT compatible)
  message(FATAL_ERROR "Installed cu130 must remain reusable on CUDA 13.1")
endif()
if(NOT reason STREQUAL "")
  message(FATAL_ERROR "Accepted builds must not retain a rejection reason: ${reason}")
endif()
