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

#ifndef QC_BUFFER__QC_BUFFER_API_HPP_
#define QC_BUFFER__QC_BUFFER_API_HPP_

#include <cstdint>
#include <memory>
#include <utility>

#include "qc_buffer/qc_buffer.hpp"
#include "qc_buffer/qc_buffer_impl.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace qc_buffer_backend
{

/// \brief Allocate a fresh qc-backed rosidl::Buffer<uint8_t> of \p count
/// bytes. Pure allocation: no data is copied. The caller owns the returned
/// buffer and assigns it wherever the message schema needs it. Throws
/// QcError if the rpcmem allocation fails.
inline rosidl::Buffer<uint8_t> allocate_buffer(size_t count)
{
  return rosidl::Buffer<uint8_t>(
    std::make_unique<QcBufferImpl<uint8_t>>(count));
}

namespace detail
{

inline QcBufferImpl<uint8_t> * qc_impl_of(rosidl::Buffer<uint8_t> & buffer)
{
  return dynamic_cast<QcBufferImpl<uint8_t> *>(buffer.get_impl());
}

inline const QcBufferImpl<uint8_t> * qc_impl_of(const rosidl::Buffer<uint8_t> & buffer)
{
  return dynamic_cast<const QcBufferImpl<uint8_t> *>(buffer.get_impl());
}

}  // namespace detail

/// \brief Return the dma-buf fd backing \p buffer, or -1 if the buffer is not
/// qc-backed (e.g. a CPU buffer) or has no allocation.
///
/// This is the integration point for HTP zero-copy: the application passes
/// this fd to the HTP accelerator so it reads the same physical memory the
/// CPU wrote, without a copy. The backend itself stays independent of any
/// accelerator SDK.
inline int get_dmabuf_fd(const rosidl::Buffer<uint8_t> & buffer)
{
  const auto * impl = detail::qc_impl_of(buffer);
  if (!impl) {
    return -1;
  }
  const auto & qc = impl->get_qc_buffer();
  return qc ? qc->dmabuf_fd() : -1;
}

/// \brief Return the CPU-accessible pointer backing \p buffer, or nullptr if
/// the buffer is not qc-backed. The bytes are directly readable/writable.
inline uint8_t * get_data_ptr(rosidl::Buffer<uint8_t> & buffer)
{
  auto * impl = detail::qc_impl_of(buffer);
  if (!impl) {
    return nullptr;
  }
  auto & qc = impl->get_qc_buffer();
  return qc ? qc->data() : nullptr;
}

}  // namespace qc_buffer_backend

#endif  // QC_BUFFER__QC_BUFFER_API_HPP_
