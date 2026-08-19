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

#include <ATen/dlpack.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace py = pybind11;

namespace
{

struct DlpackContext
{
  py::object owner;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::optional<cuda_buffer_backend::ReadHandle> read_handle;
  std::optional<cuda_buffer_backend::WriteHandle> write_handle;
};

void dlpack_deleter(DLManagedTensor * tensor)
{
  if (!tensor) {
    return;
  }
  py::gil_scoped_acquire gil;
  delete static_cast<DlpackContext *>(tensor->manager_ctx);
  delete tensor;
}

cudaStream_t stream_from_python(const py::object & stream)
{
  if (stream.is_none()) {
    return cuda_buffer_backend::get_internal_stream();
  }
  return reinterpret_cast<cudaStream_t>(stream.cast<uintptr_t>());
}

rosidl::Buffer<uint8_t> * unwrap_buffer(const py::object & buffer)
{
  py::module_ rosidl_buffer_module = py::module_::import(
    "rosidl_buffer._rosidl_buffer_py");
  uintptr_t ptr = rosidl_buffer_module.attr("_get_buffer_ptr")(buffer).cast<uintptr_t>();
  auto * cpp_buffer = reinterpret_cast<rosidl::Buffer<uint8_t> *>(ptr);
  if (!cpp_buffer || cpp_buffer->get_backend_type() != "cuda") {
    throw std::runtime_error(
            "torch_conversions: DLPack extension requires a CUDA-backed Buffer");
  }
  return cpp_buffer;
}

py::capsule make_dlpack(
  py::object buffer,
  std::vector<int64_t> shape,
  std::vector<int64_t> strides,
  uint8_t dtype_code,
  uint8_t dtype_bits,
  uint16_t dtype_lanes,
  uint64_t byte_offset,
  py::object stream,
  bool writable)
{
  auto * cpp_buffer = unwrap_buffer(buffer);
  auto * cuda_impl =
    dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(cpp_buffer->get_impl());
  if (!cuda_impl) {
    throw std::runtime_error(
            "torch_conversions: cuda backend is not a CudaBufferImpl");
  }

  auto context = std::make_unique<DlpackContext>();
  context->owner = std::move(buffer);
  context->shape = std::move(shape);
  context->strides = std::move(strides);

  cudaStream_t cuda_stream = stream_from_python(stream);
  void * data = nullptr;
  if (writable) {
    context->write_handle.emplace(
      cuda_buffer_backend::from_output_buffer(*cpp_buffer, cuda_stream));
    data = context->write_handle->get_ptr();
  } else {
    context->read_handle.emplace(
      cuda_buffer_backend::from_input_buffer(*cpp_buffer, cuda_stream));
    data = const_cast<uint8_t *>(context->read_handle->get_ptr());
  }

  auto tensor = std::make_unique<DLManagedTensor>();
  tensor->dl_tensor.data = static_cast<uint8_t *>(data) + byte_offset;
  tensor->dl_tensor.device = DLDevice{kDLCUDA, cuda_impl->get_device_id()};
  tensor->dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
  tensor->dl_tensor.dtype = DLDataType{dtype_code, dtype_bits, dtype_lanes};
  tensor->dl_tensor.shape = context->shape.data();
  tensor->dl_tensor.strides = context->strides.empty() ?
    nullptr : context->strides.data();
  tensor->dl_tensor.byte_offset = 0;
  tensor->manager_ctx = context.release();
  tensor->deleter = dlpack_deleter;

  return py::capsule(
    tensor.release(), "dltensor",
    [](PyObject * capsule) {
      if (!PyCapsule_IsValid(capsule, "dltensor")) {
        return;
      }
      auto * managed = static_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(capsule, "dltensor"));
      if (managed && managed->deleter) {
        managed->deleter(managed);
      }
    });
}

}  // namespace

PYBIND11_MODULE(_torch_conversions_py, module)
{
  module.doc() = "DLPack bindings for CUDA-backed tensor messages";

  module.def(
    "_from_input_dlpack",
    [](py::object buffer, std::vector<int64_t> shape, std::vector<int64_t> strides,
    uint8_t dtype_code, uint8_t dtype_bits, uint16_t dtype_lanes,
    uint64_t byte_offset, py::object stream)
    {
      return make_dlpack(
        std::move(buffer), std::move(shape), std::move(strides),
        dtype_code, dtype_bits, dtype_lanes, byte_offset, std::move(stream), false);
    },
    py::arg("buffer"), py::arg("shape"), py::arg("strides"),
    py::arg("dtype_code"), py::arg("dtype_bits"), py::arg("dtype_lanes"),
    py::arg("byte_offset"), py::arg("stream"));

  module.def(
    "_from_output_dlpack",
    [](py::object buffer, std::vector<int64_t> shape, std::vector<int64_t> strides,
    uint8_t dtype_code, uint8_t dtype_bits, uint16_t dtype_lanes,
    uint64_t byte_offset, py::object stream)
    {
      return make_dlpack(
        std::move(buffer), std::move(shape), std::move(strides),
        dtype_code, dtype_bits, dtype_lanes, byte_offset, std::move(stream), true);
    },
    py::arg("buffer"), py::arg("shape"), py::arg("strides"),
    py::arg("dtype_code"), py::arg("dtype_bits"), py::arg("dtype_lanes"),
    py::arg("byte_offset"), py::arg("stream"));
}
