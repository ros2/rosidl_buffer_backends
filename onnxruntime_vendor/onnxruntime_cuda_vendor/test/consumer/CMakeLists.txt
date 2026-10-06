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
project(onnxruntime_vendor_selection_test LANGUAGES CXX)

find_package(onnxruntime_core_vendor REQUIRED)
find_package(onnxruntime_cuda_vendor REQUIRED)
get_target_property(selected_library onnxruntime::onnxruntime IMPORTED_LOCATION)
if(NOT selected_library STREQUAL "${onnxruntime_cuda_vendor_LIBRARY_DIRS}/libonnxruntime.so")
  message(FATAL_ERROR "The CUDA vendor retained the CPU runtime target")
endif()
