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
#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "rosidl_buffer/buffer.hpp"

namespace py = pybind11;

// A bounded native stream gate: a Python thread must release it while the
// binding waits on CUDA. The CUDA callback never accesses Python or calls CUDA.
class StreamGate
{
  struct State
  {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered{false};
    bool released{false};
    bool timed_out{false};
  };
  std::shared_ptr<State> state_{std::make_shared<State>()};

public:
  explicit StreamGate(uintptr_t stream)
  {
    auto state = std::make_unique<std::shared_ptr<State>>(state_);
    const auto status = cudaLaunchHostFunc(
        reinterpret_cast<cudaStream_t>(stream), [](void * pointer) {
        std::unique_ptr<std::shared_ptr<State>> owner(
            static_cast<std::shared_ptr<State> *>(pointer));
        auto & state = **owner;
        std::unique_lock<std::mutex> lock(state.mutex);
        state.entered = true;
        state.condition.notify_all();
        state.timed_out = !state.condition.wait_for(
            lock, std::chrono::seconds(5), [&state] {return state.released;});
        }, state.get());
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
    state.release();
  }

  bool wait_started()
  {
    std::unique_lock<std::mutex> lock(state_->mutex);
    return state_->condition.wait_for(
      lock, std::chrono::seconds(2), [this] {return state_->entered;});
  }

  void release()
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->released = true;
    state_->condition.notify_all();
  }

  bool timed_out()
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->timed_out;
  }
};

// Test-only CPU Buffer construction; never installed.
PYBIND11_MODULE(_cuda_buffer_test_utils, module)
{
  py::class_<StreamGate>(module, "StreamGate")
  .def(py::init<uintptr_t>())
  .def("wait_started", &StreamGate::wait_started, py::call_guard<py::gil_scoped_release>())
  .def("release", &StreamGate::release)
  .def_property_readonly("timed_out", &StreamGate::timed_out);

  module.def("make_cpu_buffer", [](py::bytes data) {
      const auto * begin = reinterpret_cast<const uint8_t *>(PyBytes_AS_STRING(data.ptr()));
      auto buffer = std::make_unique<rosidl::Buffer<uint8_t>>(
        std::vector<uint8_t>(begin, begin + py::len(data)));
      py::module_ bindings = py::module_::import("rosidl_buffer._rosidl_buffer_py");
      return bindings.attr("_take_buffer_from_ptr")(
        reinterpret_cast<uintptr_t>(buffer.release()));
    });
}
