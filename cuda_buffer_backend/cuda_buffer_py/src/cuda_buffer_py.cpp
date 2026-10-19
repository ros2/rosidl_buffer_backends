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

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace py = pybind11;

namespace
{

py::object wrap_cuda_buffer(std::unique_ptr<rosidl::Buffer<uint8_t>> buffer)
{
  py::module_ rosidl_buffer_module = py::module_::import("rosidl_buffer._rosidl_buffer_py");
  return rosidl_buffer_module.attr("_take_buffer_from_ptr")(
    reinterpret_cast<uintptr_t>(buffer.release()));
}

py::object allocate_python_buffer(size_t size)
{
  return wrap_cuda_buffer(std::make_unique<rosidl::Buffer<uint8_t>>(
      cuda_buffer_backend::allocate_buffer(size)));
}

rosidl::Buffer<uint8_t> * buffer_from_python(const py::object & buffer)
{
  py::module_ module = py::module_::import("rosidl_buffer._rosidl_buffer_py");
  uintptr_t ptr = module.attr("_get_buffer_ptr")(buffer).cast<uintptr_t>();
  return reinterpret_cast<rosidl::Buffer<uint8_t> *>(ptr);
}

class CudaReadHandleWrapper
{
public:
  CudaReadHandleWrapper(
    cuda_buffer_backend::ReadHandle && handle,
    py::object owner, bool promoted = false)
  : owner_(std::move(owner)), handle_(std::move(handle)), promoted_(promoted)
  {
  }

  CudaReadHandleWrapper(const CudaReadHandleWrapper &) = delete;
  CudaReadHandleWrapper & operator=(const CudaReadHandleWrapper &) = delete;
  CudaReadHandleWrapper(CudaReadHandleWrapper &&) = default;
  // Memberwise assignment could release owner_ before the old handle records its event.
  CudaReadHandleWrapper & operator=(CudaReadHandleWrapper &&) = delete;

  uintptr_t get_ptr() const
  {
    if (!handle_) {
      throw std::runtime_error("CudaReadHandle is closed");
    }
    return reinterpret_cast<uintptr_t>(handle_->get_ptr());
  }

  bool is_closed() const {return !handle_.has_value();}

  py::object get_promoted_buffer() const
  {
    (void)get_ptr();
    return promoted_ ? owner_ : py::none();
  }

  void close()
  {
    handle_.reset();
    owner_ = py::none();
  }

private:
  // Keep the Python Buffer alive until after ReadHandle records its read event.
  py::object owner_;
  std::optional<cuda_buffer_backend::ReadHandle> handle_;
  bool promoted_;
};

class CudaWriteHandleWrapper
{
public:
  CudaWriteHandleWrapper(
    cuda_buffer_backend::WriteHandle && handle,
    py::object owner, size_t size, cudaStream_t stream, bool promoted = false)
  : owner_(std::move(owner)), handle_(std::move(handle)), size_(size), stream_(stream),
    promoted_(promoted)
  {
  }

  CudaWriteHandleWrapper(const CudaWriteHandleWrapper &) = delete;
  CudaWriteHandleWrapper & operator=(const CudaWriteHandleWrapper &) = delete;
  CudaWriteHandleWrapper(CudaWriteHandleWrapper &&) = default;
  // Memberwise assignment could release the allocation before the old handle finalizes.
  CudaWriteHandleWrapper & operator=(CudaWriteHandleWrapper &&) = delete;

  uintptr_t get_ptr()
  {
    if (!handle_) {
      throw std::runtime_error("CudaWriteHandle is closed");
    }
    return reinterpret_cast<uintptr_t>(handle_->get_ptr());
  }

  bool is_closed() const {return !handle_.has_value();}

  py::object get_promoted_buffer()
  {
    (void)get_ptr();
    return promoted_ ? owner_ : py::none();
  }

  void copy_from(const py::object & source, py::ssize_t byte_count, const py::object & stream)
  {
    // Check even zero-byte copies: a closed handle must never be reused.
    (void)get_ptr();
    if (byte_count < 0) {
      throw py::value_error("byte_count must be nonnegative");
    }
    const auto size = static_cast<size_t>(byte_count);
    if (size > size_) {
      throw py::value_error("byte_count exceeds destination size");
    }
    if (!stream.is_none() && reinterpret_cast<cudaStream_t>(stream.cast<uintptr_t>()) !=
      stream_)
    {
      throw py::value_error("stream must match the write handle's stream");
    }

    if (py::isinstance<py::bytes>(source)) {
      std::string bytes = source.cast<std::string>();
      if (size > bytes.size()) {
        throw py::value_error("byte_count exceeds host source size");
      }
      cuda_buffer_backend::to_buffer(
        bytes.data(), size, *handle_, stream_, cudaMemcpyHostToDevice);
      if (size > 0) {
        // cudaMemcpyAsync may still reference bytes after enqueueing. Keep
        // this storage alive until the host-to-device transfer completes.
        CUDA_CHECK(cudaStreamSynchronize(stream_));
      }
    } else if (py::isinstance<py::int_>(source)) {
      const auto ptr = source.cast<uintptr_t>();
      if (ptr == 0 && size > 0) {
        throw py::value_error("device source pointer must be nonzero");
      }
      cuda_buffer_backend::to_buffer(
        reinterpret_cast<const void *>(ptr), size, *handle_, stream_, cudaMemcpyDeviceToDevice);
    } else {
      throw py::type_error("source must be bytes or an integer device pointer");
    }
  }

  void close()
  {
    handle_.reset();
    owner_ = py::none();
  }

private:
  // Declared before handle_ so the WriteHandle is destroyed first and can
  // record its event while the underlying CUDA allocation is still alive.
  py::object owner_;
  std::optional<cuda_buffer_backend::WriteHandle> handle_;
  size_t size_;
  cudaStream_t stream_;
  bool promoted_;
};

cudaStream_t stream_from_python(const py::object & stream)
{
  if (stream.is_none()) {
    return cuda_buffer_backend::get_internal_stream();
  }
  return reinterpret_cast<cudaStream_t>(stream.cast<uintptr_t>());
}

CudaWriteHandleWrapper promote_output(size_t size, cudaStream_t stream)
{
  py::object buffer = allocate_python_buffer(size);
  auto * cpp_buffer = buffer_from_python(buffer);
  auto handle = cuda_buffer_backend::from_output_buffer(*cpp_buffer, stream);
  return CudaWriteHandleWrapper(std::move(handle), std::move(buffer), size, stream, true);
}

CudaReadHandleWrapper promote_input(const void * data, size_t size, cudaStream_t stream)
{
  if (size == 0) {
    throw cuda_buffer_backend::CudaError("from_input_buffer called on empty buffer");
  }
  // Python owns the replacement before any handle is acquired. The native
  // handles see CUDA storage; the wrapper records that the binding promoted it.
  py::object buffer = allocate_python_buffer(size);
  auto * cpp_buffer = buffer_from_python(buffer);
  {
    auto writer = cuda_buffer_backend::from_output_buffer(*cpp_buffer, stream);
    cuda_buffer_backend::to_buffer(data, size, writer, stream, cudaMemcpyHostToDevice);
  }
  auto reader = cuda_buffer_backend::from_input_buffer(*cpp_buffer, stream);
  // The caller's CPU storage must remain alive until the transfer completes.
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return CudaReadHandleWrapper(std::move(reader), std::move(buffer), true);
}

}  // namespace

PYBIND11_MODULE(_cuda_buffer_py, module)
{
  module.doc() = "Python bindings for creating CUDA-backed rosidl buffers";

  py::class_<CudaReadHandleWrapper>(module, "CudaReadHandle")
  .def_property_readonly("device_ptr", &CudaReadHandleWrapper::get_ptr)
  .def_property_readonly("closed", &CudaReadHandleWrapper::is_closed)
  .def("get_ptr", &CudaReadHandleWrapper::get_ptr)
  .def("get_promoted_buffer", &CudaReadHandleWrapper::get_promoted_buffer)
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
  .def_property_readonly("closed", &CudaWriteHandleWrapper::is_closed)
  .def("get_ptr", &CudaWriteHandleWrapper::get_ptr)
  .def("get_promoted_buffer", &CudaWriteHandleWrapper::get_promoted_buffer)
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
    "_allocate_buffer", &allocate_python_buffer,
    py::arg("size"));

  module.def(
    "_to_buffer",
    [](py::object source, py::ssize_t byte_count, CudaWriteHandleWrapper & handle,
    py::object stream) {
      handle.copy_from(source, byte_count, stream);
    },
    py::arg("source"), py::arg("byte_count"), py::arg("handle"),
    py::arg("stream") = py::none());

  module.def(
    "_from_output_buffer",
    [](py::object buffer, py::object stream) {
      auto * cpp_buffer = buffer_from_python(buffer);
      const auto cuda_stream = stream_from_python(stream);
      if (!dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
        cpp_buffer->get_impl()))
      {
        return promote_output(cpp_buffer->size(), cuda_stream);
      }
      auto write_handle = cuda_buffer_backend::from_output_buffer(*cpp_buffer, cuda_stream);
      return CudaWriteHandleWrapper(
        std::move(write_handle), std::move(buffer), cpp_buffer->size(), cuda_stream);
    },
    py::arg("buffer"),
    py::arg("stream") = py::none());

  module.def(
    "_from_input_buffer",
    [](py::object buffer, py::object stream) {
      const auto * cpp_buffer = buffer_from_python(buffer);
      const auto cuda_stream = stream_from_python(stream);
      if (!dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
        cpp_buffer->get_impl()))
      {
        return promote_input(cpp_buffer->data(), cpp_buffer->size(), cuda_stream);
      }
      auto read_handle = cuda_buffer_backend::from_input_buffer(*cpp_buffer, cuda_stream);
      return CudaReadHandleWrapper(std::move(read_handle), std::move(buffer));
    },
    py::arg("buffer"),
    py::arg("stream") = py::none());

  module.def(
    "_from_output_cpu_data",
    [](py::bytes data, py::object stream) {
      return promote_output(static_cast<size_t>(py::len(data)), stream_from_python(stream));
    },
    py::arg("data"),
    py::arg("stream") = py::none());

  module.def(
    "_from_input_cpu_data",
    [](py::bytes data, py::object stream) {
      return promote_input(
        PyBytes_AS_STRING(data.ptr()), static_cast<size_t>(py::len(data)),
        stream_from_python(stream));
    },
    py::arg("data"),
    py::arg("stream") = py::none());
}
