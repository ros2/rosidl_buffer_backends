// Copyright 2026 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "rosidl_buffer/buffer.hpp"

namespace py = pybind11;

// Test-only CPU Buffer construction; never installed.
PYBIND11_MODULE(_cuda_buffer_test_utils, module)
{
  module.def("make_cpu_buffer", [](py::bytes data) {
      const auto * begin = reinterpret_cast<const uint8_t *>(PyBytes_AS_STRING(data.ptr()));
      auto buffer = std::make_unique<rosidl::Buffer<uint8_t>>(
        std::vector<uint8_t>(begin, begin + py::len(data)));
      py::module_ bindings = py::module_::import("rosidl_buffer._rosidl_buffer_py");
      return bindings.attr("_take_buffer_from_ptr")(
        reinterpret_cast<uintptr_t>(buffer.release()));
    });
}
