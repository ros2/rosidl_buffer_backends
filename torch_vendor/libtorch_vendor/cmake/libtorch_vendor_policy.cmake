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

set(LIBTORCH_VENDOR_VERSION "2.9.1")
set(LIBTORCH_VENDOR_SUPPORTED_VARIANTS
  cu126 cu128 cu130)

function(libtorch_vendor_validate_variant variant)
  list(FIND LIBTORCH_VENDOR_SUPPORTED_VARIANTS
    "${variant}" variant_index)
  if(variant_index EQUAL -1)
    message(FATAL_ERROR
      "Unsupported CUDA LibTorch variant '${variant}'. Supported variants: "
      "${LIBTORCH_VENDOR_SUPPORTED_VARIANTS}")
  endif()
endfunction()

# Selects a published archive, which is not the same as requiring that
# toolkit: the archive carries its own CUDA runtime and resolves to it by
# RPATH, so the host toolkit only has to be the same major. Within a major,
# CUDA minor version compatibility lets an archive built against a later minor
# run on an earlier driver, so toolkits outside the published minors fall
# through to the nearest variant rather than failing.
function(libtorch_vendor_variant_for_cuda cuda_version output_variable)
  if(cuda_version VERSION_GREATER_EQUAL "14.0")
    message(FATAL_ERROR
      "CUDA Toolkit ${cuda_version} is a newer major than the cu130 LibTorch "
      "runtime")
  elseif(cuda_version VERSION_GREATER_EQUAL "13.0")
    set(variant cu130)
  elseif(cuda_version VERSION_GREATER_EQUAL "12.8")
    set(variant cu128)
  elseif(cuda_version VERSION_GREATER_EQUAL "12.0")
    set(variant cu126)
  else()
    message(FATAL_ERROR
      "CUDA Toolkit ${cuda_version} is unsupported; CUDA 12.0 or newer is required")
  endif()
  set(${output_variable} "${variant}" PARENT_SCOPE)
endfunction()
