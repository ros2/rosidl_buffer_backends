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
#include <utility>

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
    py::object buffer)
  : buffer_(std::move(buffer)), handle_(std::move(handle))
  {
  }

  CudaReadHandleWrapper(const CudaReadHandleWrapper &) = delete;
  CudaReadHandleWrapper & operator=(const CudaReadHandleWrapper &) = delete;
  CudaReadHandleWrapper(CudaReadHandleWrapper &&) = default;
  // Deleted rather than defaulted: the generated assignment runs in member
  // declaration order, so it would drop the Python buffer reference -- and
  // possibly the CUDA allocation with it -- before the replaced ReadHandle
  // records its read event into that same allocation.
  CudaReadHandleWrapper & operator=(CudaReadHandleWrapper &&) = delete;

  uintptr_t get_ptr() const
  {
    if (!handle_) {
      throw std::runtime_error("CudaReadHandle is closed");
    }
    return reinterpret_cast<uintptr_t>(handle_->get_ptr());
  }

  bool is_closed() const {return !handle_.has_value();}

  py::object get_buffer() const {return buffer_;}

  void close()
  {
    // Keep buffer_ available after close: when a non-CUDA input was promoted
    // this handle holds the only reference to the CUDA-backed replacement,
    // which callers may want to forward on.
    handle_.reset();
  }

private:
  // Declared before handle_ so the ReadHandle is destroyed first and can
  // record its read event while the underlying CUDA allocation is still alive.
  py::object buffer_;
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
  // Deleted rather than defaulted: the generated assignment runs in member
  // declaration order, so it would drop the Python buffer reference -- and
  // possibly the CUDA allocation with it -- before the replaced WriteHandle
  // records its write event into that same allocation.
  CudaWriteHandleWrapper & operator=(CudaWriteHandleWrapper &&) = delete;

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
  py::object wrapped = rosidl_buffer_module.attr("_take_buffer_from_ptr")(
    reinterpret_cast<uintptr_t>(buffer.get()));
  // Release only once the call above has succeeded, so a failure there does
  // not leak the buffer.
  buffer.release();
  return wrapped;
}

rosidl::Buffer<uint8_t> * buffer_ptr_from_python(const py::object & buffer)
{
  py::module_ rosidl_buffer_module = py::module_::import("rosidl_buffer._rosidl_buffer_py");
  uintptr_t ptr = rosidl_buffer_module.attr("_get_buffer_ptr")(buffer).cast<uintptr_t>();
  return reinterpret_cast<rosidl::Buffer<uint8_t> *>(ptr);
}

/// Move the CUDA buffer that from_input_buffer()/from_output_buffer()
/// allocated to promote a non-CUDA source out of \p handle and into a Python
/// Buffer. Returns none when no promotion happened. Ownership must move to
/// Python: the promoted allocation is otherwise reachable only from C++ and
/// dies with the handle, silently discarding anything written through it.
template<typename HandleT>
py::object take_promoted_buffer(HandleT & handle)
{
  auto promoted = handle.get_promoted_buffer();
  if (!promoted) {
    return py::none();
  }
  // Wrap an empty shell first: if that throws, the handle still owns the
  // promoted buffer and unwinds normally.
  auto shell = std::make_unique<rosidl::Buffer<uint8_t>>();
  auto * shell_ptr = shell.get();
  py::object wrapped = wrap_cuda_buffer(std::move(shell));
  // Only the implementation pointer moves; the CudaBufferImpl object itself
  // stays put, so the raw pointers the handle holds into it stay valid.
  *shell_ptr = std::move(*promoted);
  promoted.reset();
  handle.set_promoted_buffer({});
  return wrapped;
}

/// Allocate a CUDA buffer and enqueue a host-to-device copy of \p size bytes
/// from \p data on \p stream. Does not synchronize; the write handle records
/// the producer event as it goes out of scope so later reads order after it.
std::unique_ptr<rosidl::Buffer<uint8_t>> allocate_and_copy_async(
  const void * data, size_t size, cudaStream_t stream)
{
  auto buffer = std::make_unique<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(size));

  if (size > 0) {
    auto write_handle = cuda_buffer_backend::from_output_buffer(*buffer, stream);
    cuda_buffer_backend::to_buffer(
      data, size, write_handle, stream, cudaMemcpyHostToDevice);
  }

  return buffer;
}

std::unique_ptr<rosidl::Buffer<uint8_t>> allocate_zeroed_async(
  size_t size, cudaStream_t stream)
{
  auto buffer = std::make_unique<rosidl::Buffer<uint8_t>>(
    cuda_buffer_backend::allocate_buffer(size));

  if (size > 0) {
    auto write_handle = cuda_buffer_backend::from_output_buffer(*buffer, stream);
    CUDA_CHECK(cudaMemsetAsync(write_handle.get_ptr(), 0, size, stream));
  }

  return buffer;
}

/// Borrow the contents of \p data without copying. Accepting the buffer
/// protocol rather than py::bytes lets bytearray and memoryview inputs reach
/// the device without an intermediate host copy. The returned view keeps the
/// exporting object alive and stays valid until it is destroyed, which must
/// happen with the GIL held.
py::buffer_info byte_view(const py::buffer & data)
{
  py::buffer_info info = data.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("expected a one-dimensional buffer");
  }
  if (info.size > 1 && info.strides[0] != info.itemsize) {
    throw std::invalid_argument("expected a contiguous buffer");
  }
  return info;
}

size_t byte_view_size(const py::buffer_info & info)
{
  return static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
}

}  // namespace

PYBIND11_MODULE(_cuda_buffer_py, module)
{
  module.doc() = "Python bindings for creating CUDA-backed rosidl buffers";

  py::class_<CudaReadHandleWrapper>(module, "CudaReadHandle")
  .def_property_readonly("device_ptr", &CudaReadHandleWrapper::get_ptr)
  .def_property_readonly("buffer", &CudaReadHandleWrapper::get_buffer)
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
    [](const py::buffer & data) {
      py::buffer_info view = byte_view(data);
      const size_t size = byte_view_size(view);

      std::unique_ptr<rosidl::Buffer<uint8_t>> buffer;
      {
        // The copy and the synchronize below can take milliseconds; holding
        // the GIL across them would stall every other Python thread,
        // including the rclpy executor. `view` keeps the source alive.
        py::gil_scoped_release unlock;
        cudaStream_t stream = cuda_buffer_backend::get_internal_stream();
        buffer = allocate_and_copy_async(view.ptr, size, stream);
        if (size > 0) {
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
      }
      return wrap_cuda_buffer(std::move(buffer));
    },
    py::arg("data"));

  module.def(
    "_from_size",
    [](size_t size) {
      std::unique_ptr<rosidl::Buffer<uint8_t>> buffer;
      {
        py::gil_scoped_release unlock;
        cudaStream_t stream = cuda_buffer_backend::get_internal_stream();
        buffer = allocate_zeroed_async(size, stream);
        if (size > 0) {
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
      }
      return wrap_cuda_buffer(std::move(buffer));
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
      auto * cpp_buffer = buffer_ptr_from_python(buffer);
      auto write_handle = cuda_buffer_backend::from_output_buffer(
        *cpp_buffer, stream_from_python(stream));
      py::object promoted = take_promoted_buffer(write_handle);
      if (!promoted.is_none()) {
        buffer = std::move(promoted);
      }
      return CudaWriteHandleWrapper(std::move(write_handle), std::move(buffer));
    },
    py::arg("buffer"),
    py::arg("stream") = py::none());

  module.def(
    "_from_input_buffer",
    [](py::object buffer, py::object stream) {
      auto * cpp_buffer = buffer_ptr_from_python(buffer);
      auto read_handle = cuda_buffer_backend::from_input_buffer(
        *cpp_buffer, stream_from_python(stream));
      py::object promoted = take_promoted_buffer(read_handle);
      if (!promoted.is_none()) {
        buffer = std::move(promoted);
      }
      return CudaReadHandleWrapper(std::move(read_handle), std::move(buffer));
    },
    py::arg("buffer"),
    py::arg("stream") = py::none());

  module.def(
    "_from_input_cpu_data",
    [](const py::buffer & data, py::object stream) {
      py::buffer_info view = byte_view(data);

      cudaStream_t cuda_stream = stream_from_python(stream);
      py::object buffer = wrap_cuda_buffer(
        allocate_and_copy_async(view.ptr, byte_view_size(view), cuda_stream));
      auto read_handle = cuda_buffer_backend::from_input_buffer(
        *buffer_ptr_from_python(buffer), cuda_stream);
      return CudaReadHandleWrapper(std::move(read_handle), std::move(buffer));
    },
    py::arg("data"),
    py::arg("stream") = py::none());
}
