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

#include "qc_buffer/qc_buffer.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <utility>

#include "qc_buffer/qc_error.hpp"
#include "qc_buffer/rpcmem_loader.hpp"

namespace qc_buffer_backend
{

namespace
{

uint64_t next_uid()
{
  static std::atomic<uint64_t> counter{1};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

QcBuffer::QcBuffer(size_t byte_size)
{
  if (byte_size == 0) {
    return;
  }

  auto & loader = RpcMemLoader::instance();
  if (!loader.available()) {
    throw QcError(
            "qc buffer allocation requested but rpcmem (libcdsprpc.so) is unavailable");
  }

  ptr_ = static_cast<uint8_t *>(
    loader.alloc(kRpcmemHeapIdSystem, kRpcmemDefaultFlags, byte_size));
  if (ptr_ == nullptr) {
    throw QcError("rpcmem_alloc failed for " + std::to_string(byte_size) + " bytes");
  }

  fd_ = loader.to_fd(ptr_);
  if (fd_ < 0) {
    loader.free(ptr_);
    ptr_ = nullptr;
    throw QcError("rpcmem_to_fd failed");
  }

  size_ = byte_size;
  uid_ = next_uid();
  origin_ = Origin::kRpcmem;
}

QcBuffer::QcBuffer(uint8_t * ptr, size_t size, int fd)
: ptr_(ptr), fd_(fd), size_(size), uid_(next_uid()), origin_(Origin::kMmap)
{
}

QcBuffer::~QcBuffer()
{
  reset();
}

QcBuffer::QcBuffer(QcBuffer && other) noexcept
: ptr_(other.ptr_), fd_(other.fd_), size_(other.size_),
  uid_(other.uid_), origin_(other.origin_)
{
  other.ptr_ = nullptr;
  other.fd_ = -1;
  other.size_ = 0;
  other.uid_ = 0;
}

QcBuffer & QcBuffer::operator=(QcBuffer && other) noexcept
{
  if (this != &other) {
    reset();
    ptr_ = other.ptr_;
    fd_ = other.fd_;
    size_ = other.size_;
    uid_ = other.uid_;
    origin_ = other.origin_;
    other.ptr_ = nullptr;
    other.fd_ = -1;
    other.size_ = 0;
    other.uid_ = 0;
  }
  return *this;
}

void QcBuffer::reset() noexcept
{
  if (ptr_ != nullptr) {
    if (origin_ == Origin::kMmap) {
      ::munmap(ptr_, size_);
      if (fd_ >= 0) {
        ::close(fd_);
      }
    } else {
      RpcMemLoader::instance().free(ptr_);
    }
    ptr_ = nullptr;
  }
  fd_ = -1;
  size_ = 0;
  uid_ = 0;
}

}  // namespace qc_buffer_backend
