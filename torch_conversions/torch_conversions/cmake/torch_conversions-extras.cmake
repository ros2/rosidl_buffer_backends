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
# whichever LibTorch the consumer already resolved. libtorch_vendor is only the
# fallback for consumers that do not bring their own.
if(NOT TARGET torch)
  find_package(libtorch_vendor QUIET)
  find_package(Torch QUIET)
endif()

if(NOT TARGET torch)
  message(FATAL_ERROR
    "torch_conversions needs LibTorch. Install libtorch_vendor, or call "
    "find_package(Torch) with your own build before "
    "find_package(torch_conversions).")
endif()

set_property(TARGET torch_conversions::torch_conversions APPEND PROPERTY
  INTERFACE_LINK_LIBRARIES torch)
