// Copyright 2026 Open Source Robotics Foundation, Inc.
// Modified for the HIP port by ObeliskGate, 2026.
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

#ifndef HIP_BUFFER__HIP_BUFFER_API_HPP_
#define HIP_BUFFER__HIP_BUFFER_API_HPP_

#include <hip/hip_runtime_api.h>

#include <memory>
#include <string>
#include <utility>

#include "hip_buffer/hip_buffer.hpp"
#include "hip_buffer/hip_buffer_impl.hpp"
#include "hip_buffer/hip_error.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace hip_buffer_backend
{

/// \brief Allocate a fresh HIP-backed \c rosidl::Buffer<uint8_t> of \p count
/// bytes. Pure allocation: no handle is acquired, no data is copied. The
/// caller owns the returned buffer and assigns it wherever the schema needs
/// it (e.g. `msg.data = allocate_buffer(n)` for messages that follow the
/// `data` convention, or an arbitrary field on other messages).
inline rosidl::Buffer<uint8_t> allocate_buffer(size_t count)
{
  return rosidl::Buffer<uint8_t>(
    std::make_unique<HipBufferImpl<uint8_t>>(count));
}

namespace detail
{

/// \brief Heap-allocate a fresh HIP-backed rosidl::Buffer<uint8_t>, held via
/// shared_ptr so it can ride along with a promoted read/write handle.
inline std::shared_ptr<rosidl::Buffer<uint8_t>> allocate_hip_buffer_shared(
  size_t byte_count)
{
  auto hip_impl = std::make_unique<HipBufferImpl<uint8_t>>(byte_count);
  return std::make_shared<rosidl::Buffer<uint8_t>>(std::move(hip_impl));
}

inline HipBufferImpl<uint8_t> * hip_impl_of(rosidl::Buffer<uint8_t> & buffer)
{
  return dynamic_cast<HipBufferImpl<uint8_t> *>(buffer.get_impl());
}

}  // namespace detail

/// \brief Acquire a write handle for a HIP-backed buffer.
/// \details If \p buffer is already HIP-backed, returns a write handle
/// directly. If the buffer is non-HIP (e.g. CPU-backed), a fresh
/// HIP-backed \c rosidl::Buffer<uint8_t> is allocated (no H2D copy; the
/// caller is about to overwrite it) and a write handle for the new buffer
/// is returned. The promoted buffer is attached to the handle via
/// \c WriteHandle::get_promoted_buffer() so the caller can substitute the
/// buffer back into the message they're publishing.
template<typename T>
WriteHandle from_output_buffer(
  rosidl::Buffer<T> & buffer,
  hipStream_t stream)
{
  auto * impl = buffer.get_impl();
  if (!impl) {
    throw HipError("from_output_buffer called on buffer with null implementation");
  }
  if (buffer.size() == 0) {
    throw HipError("from_output_buffer called on empty buffer");
  }
  auto * hip_impl = dynamic_cast<HipBufferImpl<T> *>(impl);
  if (hip_impl) {
    hip_impl->set_stream(stream);
    return hip_impl->get_hip_buffer().get_write_handle(stream);
  }

  size_t byte_count = buffer.size() * sizeof(T);
  auto promoted = detail::allocate_hip_buffer_shared(byte_count);
  auto * promoted_impl = detail::hip_impl_of(*promoted);
  promoted_impl->set_stream(stream);
  auto wh = promoted_impl->get_hip_buffer().get_write_handle(stream);
  wh.set_promoted_buffer(std::move(promoted));
  return wh;
}

/// \brief Acquire a read handle for a HIP-backed buffer.
/// \details If \p buffer is already HIP-backed, returns a read handle
/// directly. If the buffer is non-HIP (e.g. CPU-backed), a new HIP-backed
/// \c rosidl::Buffer<uint8_t> is allocated, the source contents are copied
/// host-to-device, and a read handle for the new buffer is returned.
template<typename T>
ReadHandle from_input_buffer(
  const rosidl::Buffer<T> & buffer,
  hipStream_t stream)
{
  const auto * impl = buffer.get_impl();
  if (!impl) {
    throw HipError("from_input_buffer called on buffer with null implementation");
  }
  if (buffer.size() == 0) {
    throw HipError("from_input_buffer called on empty buffer");
  }
  const auto * hip_impl = dynamic_cast<const HipBufferImpl<T> *>(impl);
  if (hip_impl) {
    return hip_impl->get_hip_buffer().get_read_handle(stream);
  }

  size_t byte_count = buffer.size() * sizeof(T);
  auto promoted = detail::allocate_hip_buffer_shared(byte_count);
  auto * promoted_impl = detail::hip_impl_of(*promoted);
  {
    auto wh = promoted_impl->get_hip_buffer().get_write_handle(stream);
    HIP_CHECK(hipMemcpyAsync(
      wh.get_ptr(), buffer.data(), byte_count, hipMemcpyHostToDevice, stream));
  }
  auto rh = promoted_impl->get_hip_buffer().get_read_handle(stream);
  rh.set_promoted_buffer(std::move(promoted));
  return rh;
}

/// \brief Copy \p byte_count bytes from \p src into the memory referenced by \p wh
/// using \p stream and the given \p kind. Does not allocate.
inline void to_buffer(
  const void * src,
  size_t byte_count,
  WriteHandle & wh,
  hipStream_t stream,
  hipMemcpyKind kind = hipMemcpyDeviceToDevice)
{
  if (byte_count == 0 || !src || !wh.get_ptr()) {
    return;
  }
  HIP_CHECK(hipMemcpyAsync(wh.get_ptr(), src, byte_count, kind, stream));
}

}  // namespace hip_buffer_backend

#endif  // HIP_BUFFER__HIP_BUFFER_API_HPP_
