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
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_c_api.h"
#include "rosidl_buffer/buffer.hpp"

namespace py = pybind11;

namespace
{

using GetCudaBufferApi = cuda_buffer_status (*)(
  uint32_t, size_t, cuda_buffer_api_v1 *);

class CudaBufferApi
{
public:
  static const cuda_buffer_api_v1 & get()
  {
    static const cuda_buffer_api_v1 api = load();
    return api;
  }

private:
  static cuda_buffer_api_v1 load()
  {
    const char * override_path = std::getenv("CUDA_BUFFER_LIBRARY_PATH");
    const char * library_path =
      override_path && override_path[0] != '\0' ? override_path : "libcuda_buffer.so";
    void * library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
      const char * load_error = dlerror();
      throw std::runtime_error(
              "torch_conversions: failed to load " + std::string(library_path) +
              ": " + (load_error ? load_error : "unknown loader error"));
    }
    std::unique_ptr<void, int (*)(void *)> library_guard(library, dlclose);

    dlerror();
    void * symbol = dlsym(library, "cuda_buffer_get_api");
    const char * symbol_error = dlerror();
    if (symbol_error) {
      throw std::runtime_error(
              "torch_conversions: cuda_buffer_get_api is unavailable: " +
              std::string(symbol_error));
    }

    GetCudaBufferApi get_api = nullptr;
    static_assert(sizeof(get_api) == sizeof(symbol));
    std::memcpy(&get_api, &symbol, sizeof(get_api));

    cuda_buffer_api_v1 api{};
    cuda_buffer_status status = get_api(
      CUDA_BUFFER_C_API_VERSION, sizeof(api), &api);
    if (status != CUDA_BUFFER_STATUS_OK ||
      api.abi_version != CUDA_BUFFER_C_API_VERSION ||
      api.struct_size < sizeof(api) ||
      !api.acquire_read || !api.acquire_write || !api.release || !api.get_last_error)
    {
      throw std::runtime_error(
              "torch_conversions: installed libcuda_buffer has an incompatible C ABI");
    }
    (void)library_guard.release();
    return api;
  }
};

struct DlpackContext
{
  ~DlpackContext()
  {
    if (lease) {
      api->release(lease);
    }
  }

  py::object owner;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  const cuda_buffer_api_v1 * api{nullptr};
  cuda_buffer_lease * lease{nullptr};
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

uintptr_t stream_from_python(const py::object & stream)
{
  if (stream.is_none()) {
    return CUDA_BUFFER_STREAM_INTERNAL;
  }
  return stream.cast<uintptr_t>();
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
  const auto & api = CudaBufferApi::get();

  auto context = std::make_unique<DlpackContext>();
  context->owner = std::move(buffer);
  context->shape = std::move(shape);
  context->strides = std::move(strides);
  context->api = &api;

  uintptr_t cuda_stream = stream_from_python(stream);
  void * data = nullptr;
  int32_t device_id = -1;
  cuda_buffer_status status;
  if (writable) {
    status = api.acquire_write(
      cpp_buffer, cuda_stream, &context->lease, &data, &device_id);
  } else {
    const void * read_data = nullptr;
    status = api.acquire_read(
      cpp_buffer, cuda_stream, &context->lease, &read_data, &device_id);
    data = const_cast<void *>(read_data);
  }
  if (status != CUDA_BUFFER_STATUS_OK) {
    throw std::runtime_error(
            "torch_conversions: CUDA buffer access failed: " +
            std::string(api.get_last_error()));
  }

  auto tensor = std::make_unique<DLManagedTensor>();
  tensor->dl_tensor.data = static_cast<uint8_t *>(data) + byte_offset;
  tensor->dl_tensor.device = DLDevice{kDLCUDA, device_id};
  tensor->dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
  tensor->dl_tensor.dtype = DLDataType{dtype_code, dtype_bits, dtype_lanes};
  tensor->dl_tensor.shape = context->shape.data();
  tensor->dl_tensor.strides = context->strides.empty() ?
    nullptr : context->strides.data();
  tensor->dl_tensor.byte_offset = 0;
  tensor->manager_ctx = context.get();
  tensor->deleter = dlpack_deleter;

  py::capsule capsule(
    tensor.get(), "dltensor",
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
  (void)context.release();
  (void)tensor.release();
  return capsule;
}

}  // namespace

PYBIND11_MODULE(_cuda_dlpack_bridge, module)
{
  module.doc() = "DLPack bindings for CUDA-backed tensor messages";

  module.def(
    "_cuda_buffer_available",
    []() {
      try {
        (void)CudaBufferApi::get();
        return true;
      } catch (const std::runtime_error &) {
        return false;
      }
    });

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
