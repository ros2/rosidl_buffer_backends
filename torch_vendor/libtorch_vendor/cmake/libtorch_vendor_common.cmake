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

set(LIBTORCH_VENDOR_FALLBACK_VERSION "2.14.0")
set(LIBTORCH_VENDOR_MIN_VERSION "2.5.0")

macro(libtorch_vendor_find_cuda)
  find_package(CUDAToolkit REQUIRED)

  # Caffe2's prebuilt CUDA targets use torch::cudart.
  if(NOT TARGET torch::cudart)
    add_library(torch::cudart INTERFACE IMPORTED)
    set_property(TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES CUDA::cudart)
  endif()
endmacro()

macro(libtorch_vendor_select_sdk)
  get_filename_component(_selected_torch_root "${Torch_DIR}/../../.." REALPATH)
  if(DEFINED ENV{TORCH_INSTALL_PREFIX})
    get_filename_component(_environment_torch_root "$ENV{TORCH_INSTALL_PREFIX}" REALPATH)
    if(NOT _environment_torch_root STREQUAL _selected_torch_root)
      message(FATAL_ERROR "TORCH_INSTALL_PREFIX conflicts with the selected vendor; unset it or select that SDK when building the vendor")
    endif()
  endif()
  if(TARGET torch)
    get_target_property(_existing_torch torch IMPORTED_LOCATION_RELEASE)
    if(_existing_torch)
      get_filename_component(_existing_torch_root "${_existing_torch}/../.." REALPATH)
      if(NOT _existing_torch_root STREQUAL _selected_torch_root)
        message(FATAL_ERROR "A different Torch SDK already defined imported targets; find the selected vendor before finding Torch")
      endif()
    endif()
  endif()
  set(Caffe2_DIR "${Torch_DIR}/../Caffe2")
  set(TORCH_LIBRARY "${_selected_torch_root}/lib/libtorch.so")
  set(c10_LIBRARY "${_selected_torch_root}/lib/libc10.so")
  if(EXISTS "${_selected_torch_root}/lib/libc10_cuda.so")
    set(C10_CUDA_LIBRARY "${_selected_torch_root}/lib/libc10_cuda.so")
  endif()
endmacro()
