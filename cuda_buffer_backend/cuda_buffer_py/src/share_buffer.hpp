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

#ifndef SHARE_BUFFER_HPP_
#define SHARE_BUFFER_HPP_

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "rosidl_buffer/buffer.hpp"

namespace cuda_buffer_py
{
namespace detail
{

// Kept in a private header so the test fixture exercises the production handoff.
inline pybind11::object share_buffer(std::unique_ptr<rosidl::Buffer<uint8_t>> buffer)
{
  namespace py = pybind11;
  using SharedBuffer = std::shared_ptr<rosidl::Buffer<uint8_t>>;
  // The unique_ptr remains responsible if allocation of the control block fails.
  SharedBuffer shared(std::move(buffer));
  auto holder = std::make_unique<SharedBuffer>(std::move(shared));

  py::module_ module = py::module_::import("rosidl_buffer._rosidl_buffer_py");
  py::object helper;
  try {
    helper = module.attr("_share_buffer_from_capsule");
  } catch (py::error_already_set & error) {
    if (!error.matches(PyExc_AttributeError)) {
      throw;
    }
    throw py::import_error(
            "cuda_buffer_py requires rosidl_buffer_py with _share_buffer_from_capsule; "
            "update rosidl_buffer_py to include the shared-ownership bridge");
  }

  // String literals have static storage duration: PyCapsule does not copy names.
  // Use the PyObject* destructor overload so construction is a single PyCapsule_New.
  py::capsule capsule(
    holder.get(), "rosidl_buffer.shared_buffer", [](PyObject * object) {
      auto * pointer = PyCapsule_GetPointer(object, "rosidl_buffer.shared_buffer");
      if (!pointer) {
        PyErr_WriteUnraisable(object);
        return;
      }
      delete static_cast<SharedBuffer *>(pointer);
    });
  // No throwing operation between capsule construction and releasing the holder.
  holder.release();
  // The capsule and any receiver-created wrappers now share the same control block.
  // Unwinding is safe whether the helper has copied ownership or has not run yet.
  return helper(capsule);
}

}  // namespace detail
}  // namespace cuda_buffer_py

#endif  // SHARE_BUFFER_HPP_
