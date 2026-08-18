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

#ifndef QC_BUFFER__QC_BUFFER_IMPL_HPP_
#define QC_BUFFER__QC_BUFFER_IMPL_HPP_

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "qc_buffer/qc_buffer.hpp"
#include "qc_buffer/qc_error.hpp"
#include "rosidl_buffer/buffer_impl_base.hpp"
#include "rosidl_buffer/cpu_buffer_impl.hpp"

namespace qc_buffer_backend
{

template<typename T>
class QcBufferImpl : public rosidl::BufferImplBase<T>
{
public:
  QcBufferImpl()
  : size_(0) {}

  explicit QcBufferImpl(size_t size)
  : size_(size)
  {
    if (size_ > 0) {
      qc_buffer_ = std::make_shared<QcBuffer>(size_ * sizeof(T));
    }
  }

  /// Adopt an existing QcBuffer (same physical memory, zero-copy).
  QcBufferImpl(std::shared_ptr<QcBuffer> buffer, size_t size)
  : size_(size), qc_buffer_(std::move(buffer)) {}

  /// Cross-process subscriber path: mmap a dma-buf fd received via SCM_RIGHTS
  /// into this process's address space.
  ///
  /// Ownership: the caller transfers ownership of \p fd to this constructor.
  /// On success, fd is owned by the internal QcBuffer (closed on destruction).
  /// On failure (mmap error), fd is closed before the exception is thrown —
  /// the caller must not close fd after catching the exception.
  QcBufferImpl(int fd, size_t dmabuf_size, size_t element_count)
  : size_(element_count)
  {
    void * ptr = ::mmap(nullptr, dmabuf_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      ::close(fd);
      throw QcError(
        "QcBufferImpl: mmap of cross-process dma-buf failed: " +
        std::string(strerror(errno)));
    }
    qc_buffer_ = std::make_shared<QcBuffer>(
      static_cast<uint8_t *>(ptr), dmabuf_size, fd);
  }

  ~QcBufferImpl() override = default;

  QcBufferImpl(const QcBufferImpl &) = delete;
  QcBufferImpl & operator=(const QcBufferImpl &) = delete;
  QcBufferImpl(QcBufferImpl &&) = delete;
  QcBufferImpl & operator=(QcBufferImpl &&) = delete;

  std::string get_backend_type() const override {return "qc";}
  size_t size() const override {return size_;}

  std::unique_ptr<rosidl::BufferImplBase<T>> to_cpu() const override
  {
    auto cpu = std::make_unique<rosidl::CpuBufferImpl<T>>();
    cpu->get_storage().resize(size_);
    if (size_ > 0 && qc_buffer_ && qc_buffer_->data()) {
      std::memcpy(cpu->get_storage().data(), qc_buffer_->data(), size_ * sizeof(T));
    }
    return cpu;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> clone() const override
  {
    auto copy = std::make_unique<QcBufferImpl<T>>(size_);
    if (size_ > 0 && qc_buffer_ && qc_buffer_->data() &&
      copy->qc_buffer_ && copy->qc_buffer_->data())
    {
      std::memcpy(copy->qc_buffer_->data(), qc_buffer_->data(), size_ * sizeof(T));
    }
    return copy;
  }

  const std::shared_ptr<QcBuffer> & get_qc_buffer() const {return qc_buffer_;}
  std::shared_ptr<QcBuffer> & get_qc_buffer() {return qc_buffer_;}

private:
  size_t size_;
  std::shared_ptr<QcBuffer> qc_buffer_;
};

}  // namespace qc_buffer_backend

#endif  // QC_BUFFER__QC_BUFFER_IMPL_HPP_
