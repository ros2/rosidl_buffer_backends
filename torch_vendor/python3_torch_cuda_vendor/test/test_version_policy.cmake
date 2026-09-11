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

if(DEFINED TEST_VARIANT)
  python3_torch_cuda_vendor_validate_variant("${TEST_VARIANT}")
  return()
endif()
if(DEFINED TEST_CUDA_VERSION)
  python3_torch_cuda_vendor_variant_for_cuda("${TEST_CUDA_VERSION}" variant)
  return()
endif()

if(NOT PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION STREQUAL "2.9.1")
  message(FATAL_ERROR "Unexpected PyTorch version")
endif()
if(NOT PYTHON3_TORCH_CUDA_VENDOR_SUPPORTED_VARIANTS STREQUAL
    "cu126;cu128;cu130")
  message(FATAL_ERROR "Unexpected CUDA variants")
endif()

python3_torch_cuda_vendor_can_reuse("2.9.1" "cu130" can_reuse)
if(NOT can_reuse)
  message(FATAL_ERROR "PyTorch 2.9.1+cu130 should be reusable")
endif()
foreach(rejected_pair IN ITEMS "2.9.0,cu130" "2.9.1,cu132" "2.12.0,cu130")
  string(REPLACE "," ";" rejected_pair "${rejected_pair}")
  list(GET rejected_pair 0 version)
  list(GET rejected_pair 1 variant)
  python3_torch_cuda_vendor_can_reuse("${version}" "${variant}" can_reuse)
  if(can_reuse)
    message(FATAL_ERROR "PyTorch ${version}+${variant} must not be reused")
  endif()
endforeach()

foreach(rejected_argument IN ITEMS
    "-DTEST_VARIANT=cu132"
    "-DTEST_CUDA_VERSION=11.8"
    "-DTEST_CUDA_VERSION=14.0")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" "${rejected_argument}" -P
      "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE result
    OUTPUT_QUIET
    ERROR_QUIET)
  if(result EQUAL 0)
    message(FATAL_ERROR "${rejected_argument} should have been rejected")
  endif()
endforeach()

foreach(test_case IN ITEMS
    "12.0,cu126"
    "12.5,cu126"
    "12.6,cu126"
    "12.8,cu128"
    "12.9,cu128"
    "13.0,cu130"
    "13.2,cu130")
  string(REPLACE "," ";" test_case "${test_case}")
  list(GET test_case 0 cuda_version)
  list(GET test_case 1 expected_variant)
  python3_torch_cuda_vendor_variant_for_cuda("${cuda_version}" actual_variant)
  if(NOT actual_variant STREQUAL expected_variant)
    message(FATAL_ERROR
      "CUDA ${cuda_version}: expected ${expected_variant}, got ${actual_variant}")
  endif()
endforeach()
