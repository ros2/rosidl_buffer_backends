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
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

struct DLDevice
{
  int32_t device_type;
  int32_t device_id;
};

struct DLDataType
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensor
{
  void * data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t * shape;
  int64_t * strides;
  uint64_t byte_offset;
};

struct DLManagedTensor
{
  DLTensor dl_tensor;
  void * manager_ctx;
  void (*deleter)(DLManagedTensor *);
};

struct ManagedContext
{
  ManagedContext(std::vector<int64_t> shape_arg, py::object owner_arg)
  : shape(std::move(shape_arg)), owner(std::move(owner_arg)) {}

  std::vector<int64_t> shape;
  py::object owner;
  DLManagedTensor tensor{};
};

void delete_managed_tensor(DLManagedTensor * tensor)
{
  if (!tensor) {
    return;
  }
  auto * context = static_cast<ManagedContext *>(tensor->manager_ctx);
  py::gil_scoped_acquire gil;
  delete context;
}

py::capsule make_dlpack_capsule(
  uintptr_t data,
  int32_t device_type,
  int32_t device_id,
  uint8_t dtype_code,
  uint8_t dtype_bits,
  uint16_t dtype_lanes,
  std::vector<int64_t> shape,
  uint64_t byte_offset,
  py::object owner)
{
  auto context = std::make_unique<ManagedContext>(std::move(shape), std::move(owner));
  auto & tensor = context->tensor;
  tensor.dl_tensor.data = reinterpret_cast<void *>(data);
  tensor.dl_tensor.device = {device_type, device_id};
  tensor.dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
  tensor.dl_tensor.dtype = {dtype_code, dtype_bits, dtype_lanes};
  tensor.dl_tensor.shape = context->shape.data();
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = byte_offset;
  tensor.manager_ctx = context.get();
  tensor.deleter = delete_managed_tensor;

  auto capsule = py::capsule(
    &tensor, "dltensor",
    [](PyObject * object) {
      if (PyCapsule_IsValid(object, "dltensor")) {
        auto * managed = static_cast<DLManagedTensor *>(
          PyCapsule_GetPointer(object, "dltensor"));
        managed->deleter(managed);
      }
    });
  context.release();
  return capsule;
}

}  // namespace

PYBIND11_MODULE(_dlpack_bridge, module)
{
  module.def(
    "make_dlpack_capsule", &make_dlpack_capsule,
    py::arg("data"), py::arg("device_type"), py::arg("device_id"),
    py::arg("dtype_code"), py::arg("dtype_bits"), py::arg("dtype_lanes"),
    py::arg("shape"), py::arg("byte_offset"), py::arg("owner"));
}
