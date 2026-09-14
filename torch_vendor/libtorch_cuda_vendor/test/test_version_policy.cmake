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

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/libtorch_cuda_vendor_policy.cmake")
if(DEFINED TEST_CUDA_VERSION)
  libtorch_cuda_vendor_validate_variant_for_cuda("${TEST_VARIANT}" "${TEST_CUDA_VERSION}")
  return()
endif()
foreach(version IN ITEMS 13.1 13.1.115 13.2 13.9)
  libtorch_cuda_vendor_variant_for_cuda("${version}" variant)
  if(NOT variant STREQUAL "cu130")
    message(FATAL_ERROR "Default fallback must be cu130")
  endif()
endforeach()
libtorch_cuda_vendor_validate_variant_for_cuda(cu132 13.2)
foreach(pair IN ITEMS "cu130,12.6" "cu130,13.0" "cu130,14.0" "cu132,13.1" "cu126,13.1")
  string(REPLACE "," ";" fields "${pair}")
  list(GET fields 0 variant)
  list(GET fields 1 version)
  execute_process(COMMAND "${CMAKE_COMMAND}"
    "-DTEST_VARIANT=${variant}" "-DTEST_CUDA_VERSION=${version}" -P
    "${CMAKE_CURRENT_LIST_FILE}" RESULT_VARIABLE result OUTPUT_QUIET ERROR_VARIABLE error)
  if(result EQUAL 0)
    message(FATAL_ERROR "Incompatible fallback ${pair} must be rejected")
  endif()
endforeach()
