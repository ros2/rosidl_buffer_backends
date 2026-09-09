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

set(PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION "2.9.1")
set(PYTHON3_TORCH_CUDA_VENDOR_SUPPORTED_VARIANTS
  cu126 cu128 cu130)

function(python3_torch_cuda_vendor_validate_variant variant)
  list(FIND PYTHON3_TORCH_CUDA_VENDOR_SUPPORTED_VARIANTS
    "${variant}" variant_index)
  if(variant_index EQUAL -1)
    message(FATAL_ERROR
      "Unsupported CUDA Torch variant '${variant}'. Supported variants: "
      "${PYTHON3_TORCH_CUDA_VENDOR_SUPPORTED_VARIANTS}")
  endif()
endfunction()

function(python3_torch_cuda_vendor_can_reuse version variant output_variable)
  list(FIND PYTHON3_TORCH_CUDA_VENDOR_SUPPORTED_VARIANTS
    "${variant}" variant_index)
  if(version STREQUAL PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION AND
      NOT variant_index EQUAL -1)
    set(can_reuse TRUE)
  else()
    set(can_reuse FALSE)
  endif()
  set(${output_variable} "${can_reuse}" PARENT_SCOPE)
endfunction()

function(python3_torch_cuda_vendor_variant_for_cuda cuda_version output_variable)
  if(cuda_version VERSION_GREATER_EQUAL "13.0")
    set(variant cu130)
  elseif(cuda_version VERSION_GREATER_EQUAL "12.8")
    set(variant cu128)
  elseif(cuda_version VERSION_GREATER_EQUAL "12.6")
    set(variant cu126)
  else()
    message(FATAL_ERROR
      "CUDA Toolkit ${cuda_version} has no PyTorch 2.9.1 wheel; "
      "CUDA 12.6 or newer is required")
  endif()
  set(${output_variable} "${variant}" PARENT_SCOPE)
endfunction()
