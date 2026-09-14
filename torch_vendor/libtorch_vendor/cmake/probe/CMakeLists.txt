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
include("${CMAKE_CURRENT_LIST_DIR}/../libtorch_vendor_common.cmake")
if(EXISTS "${torch_root}/lib/libtorch_cuda.so")
  libtorch_vendor_find_cuda()
endif()
libtorch_vendor_select_sdk()
find_package(Torch REQUIRED)
if(Torch_VERSION VERSION_LESS LIBTORCH_VENDOR_MIN_VERSION)
  reject_provider("Rejected LibTorch ${Torch_VERSION}: require >=${LIBTORCH_VENDOR_MIN_VERSION}")
endif()
if(REQUIRED_TORCH_VERSION AND NOT Torch_VERSION VERSION_EQUAL REQUIRED_TORCH_VERSION)
  string(CONCAT reason "Rejected LibTorch ${Torch_VERSION}: the conversion core uses ${REQUIRED_TORCH_VERSION}; "
    "native CPU and CUDA providers must use the same release")
  reject_provider("${reason}")
endif()
set(variant cpu)
if(TARGET torch_cuda)
  set(variant cuda)
  if(EXISTS "${torch_root}/build-version")
    file(READ "${torch_root}/build-version" distribution)
  elseif(EXISTS "${torch_root}/version.py")
    file(READ "${torch_root}/version.py" distribution)
  endif()
  if(distribution MATCHES "[+](cu[0-9]+)")
    set(variant "${CMAKE_MATCH_1}")
  endif()
elseif(REQUIRE_CUDA)
  reject_provider("CUDA-enabled LibTorch is required; TorchConfig.cmake does not export torch_cuda")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compatible.txt" "${Torch_DIR}\n${variant}\n${Torch_VERSION}\n")
