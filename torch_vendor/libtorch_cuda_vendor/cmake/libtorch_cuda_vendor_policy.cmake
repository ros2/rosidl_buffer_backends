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
set(LIBTORCH_CUDA_VENDOR_SUPPORTED_VARIANTS cu130 cu132)

function(libtorch_cuda_vendor_validate_variant variant)
  list(FIND LIBTORCH_CUDA_VENDOR_SUPPORTED_VARIANTS "${variant}" index)
  if(index EQUAL -1)
    message(FATAL_ERROR "Unsupported fallback Torch variant '${variant}'; use ${LIBTORCH_CUDA_VENDOR_SUPPORTED_VARIANTS}")
  endif()
endfunction()

# The pinned fallback uses cu130 with the CUDA 13.1 toolkit baseline.
function(libtorch_cuda_vendor_variant_for_cuda cuda_version output_variable)
  if(cuda_version VERSION_LESS "13.1" OR NOT cuda_version VERSION_LESS "14")
    message(FATAL_ERROR "The pinned Torch 2.14.0 CUDA fallback requires CUDA >=13.1,<14; "
      "found ${cuda_version}. Install CUDA 13.1 through rosdep/APT and select it with "
      "-DCUDAToolkit_ROOT=/usr/local/cuda-13.1.")
  endif()
  set(${output_variable} cu130 PARENT_SCOPE)
endfunction()

function(libtorch_cuda_vendor_validate_variant_for_cuda variant cuda_version)
  libtorch_cuda_vendor_variant_for_cuda("${cuda_version}" default_variant)
  libtorch_cuda_vendor_validate_variant("${variant}")
  if(variant STREQUAL "cu132" AND cuda_version VERSION_LESS "13.2")
    message(FATAL_ERROR "Torch cu132 requires a CUDA >=13.2 toolkit; selected CUDA toolkit is "
      "${cuda_version}. Select cu130 or set CUDAToolkit_ROOT to a compatible toolkit.")
  endif()
endfunction()
