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

cmake_minimum_required(VERSION 3.20)
project(libtorch_compatibility_probe LANGUAGES CXX)
function(reject_provider reason)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rejection_reason.txt" "${reason}")
  message(FATAL_ERROR "${reason}")
endfunction()
find_path(_torch_config_dir TorchConfig.cmake
  HINTS "${Torch_DIR}" ENV Torch_DIR
  PATH_SUFFIXES Torch share/cmake/Torch)
if(NOT _torch_config_dir)
  reject_provider("No existing TorchConfig.cmake was found")
endif()
set(Torch_DIR "${_torch_config_dir}")
get_filename_component(torch_root "${Torch_DIR}/../../.." ABSOLUTE)
if(EXISTS "${torch_root}/build-version")
  file(READ "${torch_root}/build-version" distribution)
elseif(EXISTS "${torch_root}/version.py")
  file(STRINGS "${torch_root}/version.py" distribution REGEX "^__version__ =")
else()
  reject_provider("Cannot identify this LibTorch distribution's CUDA build")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../libtorch_vendor_version.cmake")
if(NOT distribution MATCHES "([0-9]+[.][0-9]+[.][0-9]+)[+](cpu|cu)")
  reject_provider("Require a stable LibTorch release with CPU or CUDA build metadata")
endif()
set(distribution_version "${CMAKE_MATCH_1}")
if(distribution_version VERSION_LESS LIBTORCH_VENDOR_MIN_VERSION)
  string(CONCAT reason "Rejected LibTorch ${distribution_version}: require "
    ">=${LIBTORCH_VENDOR_MIN_VERSION}")
  reject_provider("${reason}")
endif()
if(REQUIRED_TORCH_VERSION AND NOT distribution_version VERSION_EQUAL REQUIRED_TORCH_VERSION)
  string(CONCAT reason "Rejected LibTorch ${distribution_version}: the CPU/core provider selected "
    "${REQUIRED_TORCH_VERSION}; CPU and CUDA releases must match")
  reject_provider("${reason}")
endif()
if(distribution MATCHES "[+]cu((12|13)([0-9]+))")
  set(variant "cu${CMAKE_MATCH_1}")
  set(framework_cuda_major "${CMAKE_MATCH_2}")
  set(framework_cuda_version "${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
elseif(distribution MATCHES "[+]cpu")
  set(variant cpu)
else()
  reject_provider("Require a CPU or CUDA >=12,<14 LibTorch distribution")
endif()
if(REQUIRE_CUDA AND variant STREQUAL "cpu")
  reject_provider("A CUDA provider requires CUDA-enabled LibTorch")
endif()
if(NOT variant STREQUAL "cpu")
  include("${CMAKE_CURRENT_LIST_DIR}/../libtorch_vendor_cuda.cmake")
  set(reason "")
  if(NOT framework_cuda_major STREQUAL CUDAToolkit_VERSION_MAJOR)
    set(reason "CUDA major versions differ")
  elseif(CUDAToolkit_VERSION VERSION_LESS framework_cuda_version)
    set(reason "the Torch CUDA build is newer than the selected toolkit")
  endif()
  if(reason)
    string(CONCAT reason
      "Rejected LibTorch ${variant} (built with CUDA ${framework_cuda_version}): "
      "selected CUDA toolkit is ${CUDAToolkit_VERSION}; ${reason}. "
      "Require the same CUDA major and an older or equal Torch CUDA build minor. "
      "Select a compatible toolkit with CUDAToolkit_ROOT or a compatible LibTorch SDK with Torch_DIR.")
    reject_provider("${reason}")
  endif()
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../libtorch_vendor_select.cmake")
find_package(Torch ${distribution_version} EXACT REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(check check.cpp)
target_link_libraries(check PRIVATE ${TORCH_LIBRARIES})
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compatible.txt" "${Torch_DIR}\n${variant}\n${Torch_VERSION}\n")
