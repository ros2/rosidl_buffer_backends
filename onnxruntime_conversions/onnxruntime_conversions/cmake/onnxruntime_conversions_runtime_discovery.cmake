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

include("${ament_cmake_core_DIR}/index/ament_index_get_resource.cmake")
include("${ament_cmake_core_DIR}/index/ament_index_get_resources.cmake")

set(onnxruntime_conversions_Interface_FOUND TRUE)

if("Interface" IN_LIST onnxruntime_conversions_FIND_COMPONENTS)
  return()
endif()

ament_index_get_resources(
  _onnxruntime_conversions_runtimes
  "onnxruntime_conversions__runtime"
)
list(LENGTH _onnxruntime_conversions_runtimes
  _onnxruntime_conversions_runtime_count)
if(NOT _onnxruntime_conversions_runtime_count EQUAL 1)
  message(FATAL_ERROR
    "onnxruntime_conversions requires exactly one C++ runtime package, but "
    "found ${_onnxruntime_conversions_runtime_count}: "
    "[${_onnxruntime_conversions_runtimes}]. Install/build exactly one of "
    "onnxruntime_conversions_cpu or onnxruntime_conversions_cuda, then "
    "reconfigure dependent packages.")
endif()

list(GET _onnxruntime_conversions_runtimes 0
  _onnxruntime_conversions_runtime_package)
ament_index_get_resource(
  _onnxruntime_conversions_runtime_target
  "onnxruntime_conversions__runtime"
  "${_onnxruntime_conversions_runtime_package}"
)
string(STRIP "${_onnxruntime_conversions_runtime_target}"
  _onnxruntime_conversions_runtime_target)

find_package("${_onnxruntime_conversions_runtime_package}" CONFIG REQUIRED)
if(NOT TARGET "${_onnxruntime_conversions_runtime_target}")
  message(FATAL_ERROR
    "${_onnxruntime_conversions_runtime_package} registered missing target "
    "'${_onnxruntime_conversions_runtime_target}'")
endif()

target_link_libraries(
  onnxruntime_conversions::onnxruntime_conversions
  INTERFACE "${_onnxruntime_conversions_runtime_target}"
)

unset(_onnxruntime_conversions_runtime_count)
unset(_onnxruntime_conversions_runtime_package)
unset(_onnxruntime_conversions_runtime_target)
unset(_onnxruntime_conversions_runtimes)
