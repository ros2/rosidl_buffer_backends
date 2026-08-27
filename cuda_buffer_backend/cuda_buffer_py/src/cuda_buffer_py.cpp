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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace py = pybind11;

namespace
{

class CudaReadHandleWrapper
{
public:
  CudaReadHandleWrapper(
    cuda_buffer_backend::ReadHandle && handle,
    py::object owner)
  : owner_(std::move(owner)), handle_(std::move(handle))
  {
  }

  CudaReadHandleWrapper(const CudaReadHandleWrapper &) = delete;
  CudaReadHandleWrapper & operator=(const CudaReadHandleWrapper &) = delete;
  CudaReadHandleWrapper(CudaReadHandleWrapper &&) = default;
  CudaReadHandleWrapper & operator=(CudaReadHandleWrapper &&) = default;

  uintptr_t get_ptr() const
  {
    if (!handle_) {
      throw std::runtime_error("CudaReadHandle is closed");
    }
    return reinterpret_cast<uintptr_t>(handle_->get_ptr());
  }

  bool is_closed() const {return !handle_.has_value();}

  void close()
  {
    handle_.reset();
    owner_ = py::none();
  }

private:
  // Keep the Python Buffer alive until after ReadHandle records its read event.
  py::object owner_;
  std::optional<cuda_buffer_backend::ReadHandle> handle_;
};

class CudaWriteHandleWrapper
{
public:
  CudaWriteHandleWrapper(
    cuda_buffer_backend::WriteHandle && handle,
    py::object buffer)
  : buffer_(std::move(buffer)), handle_(std::move(handle))
  {
  }

  CudaWriteHandleWrapper(const CudaWriteHandleWrapper &) = delete;
  CudaWriteHandleWrapper & operator=(const CudaWriteHandleWrapper &) = delete;
  CudaWriteHandleWrapper(CudaWriteHandleWrapper &&) = default;
  CudaWriteHandleWrapper & operator=(CudaWriteHandleWrapper &&) = default;

  uintptr_t get_ptr()
  {
    if (!handle_) {
      throw std::runtime_error("CudaWriteHandle is closed");
    }
    return reinterpret_cast<uintptr_t>(handle_->get_ptr());
  }

  bool is_closed() const {return !handle_.has_value();}

  py::object get_buffer() const {return buffer_;}

  void close()
  {
    // Keep buffer_ available after close so callers can assign a promoted
    // output buffer to their message after the write event is recorded.
    handle_.reset();
  }

private:
  // Declared before handle_ so the WriteHandle is destroyed first and can
  // record its event while the underlying CUDA allocation is still alive.
  py::object buffer_;
  std::optional<cuda_buffer_backend::WriteHandle> handle_;
};

cudaStream_t stream_from_python(const py::object & stream)
{
  if (stream.is_none()) {
    return cuda_buffer_backend::get_internal_stream();
  }
  return reinterpret_cast<cudaStream_t>(stream.cast<uintptr_t>());
}

py::object wrap_cuda_buffer(std::unique_ptr<rosidl::Buffer<uint8_t>> buffer)
{
  py::module_ rosidl_buffer_module = py::module_::import("rosidl_buffer._rosidl_buffer_py");
  return rosidl_buffer_module.attr("_take_buffer_from_ptr")(
    reinterpret_cast<uintptr_t>(buffer.release()));
}

std::unique_ptr<rosidl::Buffer<uint8_t>> allocate_and_copy(
  const void * data, size_t size)
{
  auto buffer = std::make_unique<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(size));

  if (size > 0) {
    cudaStream_t stream = cuda_buffer_backend::get_internal_stream();
    {
      auto write_handle = cuda_buffer_backend::from_output_buffer(*buffer, stream);
      cuda_buffer_backend::to_buffer(
        data, size, write_handle, stream, cudaMemcpyHostToDevice);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  return buffer;
}

std::unique_ptr<rosidl::Buffer<uint8_t>> allocate_zeroed(size_t size)
{
  auto buffer = std::make_unique<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(size));

  if (size > 0) {
    cudaStream_t stream = cuda_buffer_backend::get_internal_stream();
    {
      auto write_handle = cuda_buffer_backend::from_output_buffer(*buffer, stream);
      CUDA_CHECK(cudaMemsetAsync(write_handle.get_ptr(), 0, size, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  return buffer;
}

}  // namespace

PYBIND11_MODULE(_cuda_buffer_py, module)
{
  module.doc() = "Python bindings for creating CUDA-backed rosidl buffers";

  py::class_<CudaReadHandleWrapper>(module, "CudaReadHandle")
  .def_property_readonly("device_ptr", &CudaReadHandleWrapper::get_ptr)
  .def_property_readonly("closed", &CudaReadHandleWrapper::is_closed)
  .def("get_ptr", &CudaReadHandleWrapper::get_ptr)
  .def("close", &CudaReadHandleWrapper::close)
  .def(
    "__enter__",
    [](CudaReadHandleWrapper & self) -> CudaReadHandleWrapper & {
      (void)self.get_ptr();
      return self;
    },
    py::return_value_policy::reference_internal)
  .def(
    "__exit__",
    [](CudaReadHandleWrapper & self, py::object, py::object, py::object) {
      self.close();
      return false;
    });

  py::class_<CudaWriteHandleWrapper>(module, "CudaWriteHandle")
  .def_property_readonly("device_ptr", &CudaWriteHandleWrapper::get_ptr)
  .def_property_readonly("buffer", &CudaWriteHandleWrapper::get_buffer)
  .def_property_readonly("closed", &CudaWriteHandleWrapper::is_closed)
  .def("get_ptr", &CudaWriteHandleWrapper::get_ptr)
  .def("close", &CudaWriteHandleWrapper::close)
  .def(
    "__enter__",
    [](CudaWriteHandleWrapper & self) -> CudaWriteHandleWrapper & {
      (void)self.get_ptr();
      return self;
    },
    py::return_value_policy::reference_internal)
  .def(
    "__exit__",
    [](CudaWriteHandleWrapper & self, py::object, py::object, py::object) {
      self.close();
      return false;
    });

  module.def(
    "_from_cpu_data",
    [](py::bytes data) {
      std::string bytes = data;
      return wrap_cuda_buffer(allocate_and_copy(bytes.data(), bytes.size()));
    },
    py::arg("data"));

  module.def(
    "_from_size",
    [](size_t size) {
      return wrap_cuda_buffer(allocate_zeroed(size));
    },
    py::arg("size"));

  module.def(
    "_allocate_buffer",
    [](size_t size) {
      auto buffer = std::make_unique<rosidl::Buffer<uint8_t>>(
        cuda_buffer_backend::allocate_buffer(size));
      return wrap_cuda_buffer(std::move(buffer));
    },
    py::arg("size"));

  module.def(
    "_from_output_buffer",
    [](py::object buffer, py::object stream) {
      py::module_ rosidl_buffer_module = py::module_::import(
        "rosidl_buffer._rosidl_buffer_py");
      uintptr_t ptr = rosidl_buffer_module.attr("_get_buffer_ptr")(buffer).cast<uintptr_t>();
      auto * cpp_buffer = reinterpret_cast<rosidl::Buffer<uint8_t> *>(ptr);
      auto write_handle = cuda_buffer_backend::from_output_buffer(
        *cpp_buffer, stream_from_python(stream));
      return CudaWriteHandleWrapper(std::move(write_handle), std::move(buffer));
    },
    py::arg("buffer"),
    py::arg("stream") = py::none());

  module.def(
    "_from_input_buffer",
    [](py::object buffer, py::object stream) {
      py::module_ rosidl_buffer_module = py::module_::import(
        "rosidl_buffer._rosidl_buffer_py");
      uintptr_t ptr = rosidl_buffer_module.attr("_get_buffer_ptr")(buffer).cast<uintptr_t>();
      auto * cpp_buffer = reinterpret_cast<rosidl::Buffer<uint8_t> *>(ptr);
      auto read_handle = cuda_buffer_backend::from_input_buffer(
        *cpp_buffer, stream_from_python(stream));
      return CudaReadHandleWrapper(std::move(read_handle), std::move(buffer));
    },
    py::arg("buffer"),
    py::arg("stream") = py::none());

  module.def(
    "_from_input_cpu_data",
    [](py::bytes data, py::object stream) {
      std::string bytes = data;
      std::vector<uint8_t> storage(bytes.begin(), bytes.end());
      rosidl::Buffer<uint8_t> cpu_buffer(std::move(storage));
      auto read_handle = cuda_buffer_backend::from_input_buffer(
        cpu_buffer, stream_from_python(stream));
      return CudaReadHandleWrapper(std::move(read_handle), py::none());
    },
    py::arg("data"),
    py::arg("stream") = py::none());
}
