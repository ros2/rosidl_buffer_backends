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
#include <utility>
#include <vector>

#include "rosidl_buffer/buffer.hpp"
#include "rosidl_buffer/cpu_buffer_impl.hpp"
#include "share_buffer.hpp"

namespace py = pybind11;

struct DestructionCount
{
  size_t value = 0;
};

class TrackedCpuBuffer : public rosidl::CpuBufferImpl<uint8_t>
{
public:
  explicit TrackedCpuBuffer(std::shared_ptr<DestructionCount> count)
  : count_(std::move(count)) {}

  ~TrackedCpuBuffer() override {++count_->value;}

private:
  std::shared_ptr<DestructionCount> count_;
};

// Test-only CPU Buffer construction; never installed.
PYBIND11_MODULE(_cuda_buffer_test_utils, module)
{
  py::class_<DestructionCount, std::shared_ptr<DestructionCount>>(module, "DestructionCount")
  .def(py::init<>())
  .def_readonly("value", &DestructionCount::value);

  module.def("make_tracked_buffer", [](std::shared_ptr<DestructionCount> count) {
      auto impl = std::make_unique<TrackedCpuBuffer>(std::move(count));
      impl->get_storage() = {1, 2, 3, 4};
      return cuda_buffer_py::detail::share_buffer(
        std::make_unique<rosidl::Buffer<uint8_t>>(std::move(impl)));
    });

  module.def("make_cpu_buffer", [](py::bytes data) {
      const auto * begin = reinterpret_cast<const uint8_t *>(PyBytes_AS_STRING(data.ptr()));
      auto buffer = std::make_unique<rosidl::Buffer<uint8_t>>(
        std::vector<uint8_t>(begin, begin + py::len(data)));
      return cuda_buffer_py::detail::share_buffer(std::move(buffer));
    });
}
