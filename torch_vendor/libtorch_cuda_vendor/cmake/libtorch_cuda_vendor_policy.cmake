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

set(LIBTORCH_CUDA_VENDOR_VERSION "2.14.0")
set(LIBTORCH_CUDA_VENDOR_SUPPORTED_VARIANTS cu126 cu130 cu132)

function(libtorch_cuda_vendor_validate_variant variant)
  list(FIND LIBTORCH_CUDA_VENDOR_SUPPORTED_VARIANTS "${variant}" index)
  if(index EQUAL -1)
    message(FATAL_ERROR "Unsupported Torch variant '${variant}'; use ${LIBTORCH_CUDA_VENDOR_SUPPORTED_VARIANTS}")
  endif()
endfunction()

function(libtorch_cuda_vendor_variant_for_cuda cuda_version output_variable)
  if(cuda_version VERSION_LESS "12.6" OR NOT cuda_version VERSION_LESS "14")
    message(FATAL_ERROR "CUDA ${cuda_version} is unsupported; require CUDA >=12.6,<14")
  elseif(cuda_version VERSION_LESS "13")
    set(variant cu126)
  elseif(cuda_version VERSION_LESS "13.2")
    set(variant cu130)
  else()
    set(variant cu132)
  endif()
  set(${output_variable} "${variant}" PARENT_SCOPE)
endfunction()

# Require the same CUDA major and a Torch build minor <= the toolkit minor.
# The optional fourth argument receives the rejection reason.
function(libtorch_cuda_vendor_cuda_compatible variant cuda_version output_variable)
  set(compatible FALSE)
  set(reason "")
  if(NOT variant MATCHES "^cu(12|13)([0-9]+)$")
    string(CONCAT reason "Rejected Torch build '${variant}': require a CUDA 12 or 13 build; "
      "selected CUDA toolkit is ${cuda_version}.")
  else()
    set(framework_cuda_major "${CMAKE_MATCH_1}")
    set(framework_cuda_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    string(REGEX MATCH "^[0-9]+" toolkit_cuda_major "${cuda_version}")
    if(cuda_version VERSION_LESS "12.6" OR NOT cuda_version VERSION_LESS "14")
      set(reason "selected toolkit is outside the supported range >=12.6,<14")
    elseif(NOT framework_cuda_major STREQUAL toolkit_cuda_major)
      set(reason "CUDA major versions differ")
    elseif(cuda_version VERSION_LESS framework_cuda_version)
      set(reason "the Torch CUDA build is newer than the selected toolkit")
    else()
      set(compatible TRUE)
    endif()
    if(NOT compatible)
      string(CONCAT reason
        "Rejected Torch ${variant} (built with CUDA ${framework_cuda_version}): "
        "selected CUDA toolkit is ${cuda_version}; ${reason}. "
        "Require the same CUDA major and an older or equal Torch CUDA build minor. "
        "Select a compatible toolkit with CUDAToolkit_ROOT or a compatible Torch CUDA build.")
    endif()
  endif()
  set(${output_variable} "${compatible}" PARENT_SCOPE)
  if(ARGC GREATER 3)
    set(${ARGV3} "${reason}" PARENT_SCOPE)
  endif()
endfunction()

function(libtorch_cuda_vendor_validate_variant_for_cuda variant cuda_version)
  libtorch_cuda_vendor_validate_variant("${variant}")
  libtorch_cuda_vendor_cuda_compatible("${variant}" "${cuda_version}" compatible reason)
  if(NOT compatible)
    message(FATAL_ERROR "${reason}")
  endif()
endfunction()
