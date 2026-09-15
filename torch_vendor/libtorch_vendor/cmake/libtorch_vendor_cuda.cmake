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

find_package(CUDAToolkit 12.6 REQUIRED)
if(NOT CUDAToolkit_VERSION VERSION_LESS "14.0")
  message(FATAL_ERROR "LibTorch requires CUDA >=12.6,<14")
endif()

# Caffe2's prebuilt CUDA targets use torch::cudart.
if(NOT TARGET torch::cudart)
  add_library(torch::cudart INTERFACE IMPORTED)
  set_property(TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES CUDA::cudart)
endif()
