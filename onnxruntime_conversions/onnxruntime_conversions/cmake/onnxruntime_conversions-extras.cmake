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

# The adapter compiles in the consumer's translation units, so it binds to
# whichever ONNX Runtime the consumer already resolved. onnxruntime_cuda_vendor
# is only the fallback for consumers that do not bring their own.
if(NOT TARGET onnxruntime::onnxruntime)
  find_package(onnxruntime_cuda_vendor QUIET)
endif()
if(NOT TARGET onnxruntime::onnxruntime)
  find_package(onnxruntime QUIET)
endif()

if(NOT TARGET onnxruntime::onnxruntime)
  message(FATAL_ERROR
    "onnxruntime_conversions needs ONNX Runtime. Install "
    "onnxruntime_cuda_vendor, or call find_package(onnxruntime) with your own "
    "build before find_package(onnxruntime_conversions).")
endif()

set_property(TARGET onnxruntime_conversions::onnxruntime_conversions APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES onnxruntime::onnxruntime)
