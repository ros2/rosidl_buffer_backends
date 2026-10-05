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

#ifndef INTEL_BUFFER__INTEL_BUFFER_IMPL_HPP_
#define INTEL_BUFFER__INTEL_BUFFER_IMPL_HPP_

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include <unistd.h>
#include <sys/mman.h>

#include "intel_memory_core/intel_memory_buffer_pool.hpp"
#include "rosidl_buffer/buffer_impl_base.hpp"
#include "rosidl_buffer/cpu_buffer_impl.hpp"

namespace intel_buffer_backend
{


using intel_memory_core::PoolBlock;
using intel_memory_core::IntelMemoryBufferPool;


template<typename T>
class IntelBufferImpl : public rosidl::BufferImplBase<T>
{
public:
  IntelBufferImpl()
  : size_(0), block_(nullptr) {}

  explicit IntelBufferImpl(size_t size)
  : size_(size), block_(nullptr)
  {
    if (size_ > 0) {
      allocate(size_);
    }
  }


  IntelBufferImpl(PoolBlock * block, size_t size, bool owns_block)
  : size_(size), block_(block), owns_block_(owns_block) {}

  IntelBufferImpl(PoolBlock * block, size_t size, bool owns_block, bool standalone)
  : size_(size), block_(block), owns_block_(owns_block), standalone_(standalone) {}

  /// wrap_dmabuf(): the buffer always owns the mmap() it was created with, even
  /// when the caller keeps the fd (owns_block=false).
  IntelBufferImpl(
    PoolBlock * block, size_t size, bool owns_block, bool standalone, bool owns_mapping)
  : size_(size), block_(block), owns_block_(owns_block), standalone_(standalone),
    owns_mapping_(owns_mapping) {}

  ~IntelBufferImpl()
  {
    if (!block_) {
      return;
    }
    if (standalone_) {

      if (owns_block_) {
        free_standalone_resource(block_);
      } else if (owns_mapping_ && block_->ptr && block_->ptr != MAP_FAILED) {
        munmap(block_->ptr, block_->size);
      }
      delete block_;
    } else if (owns_block_) {
      auto pool = get_or_create_global_pool();
      if (pool) {
        pool->release(block_);
      }
    }
  }

  IntelBufferImpl(const IntelBufferImpl &) = delete;
  IntelBufferImpl & operator=(const IntelBufferImpl &) = delete;
  IntelBufferImpl(IntelBufferImpl &&) = delete;
  IntelBufferImpl & operator=(IntelBufferImpl &&) = delete;

  std::string get_backend_type() const override { return "intel_buffer"; }

  size_t size() const override { return size_; }

  std::unique_ptr<rosidl::BufferImplBase<T>> to_cpu() const override
  {
    auto cpu = std::make_unique<rosidl::CpuBufferImpl<T>>();
    cpu->get_storage().resize(size_);
    if (size_ > 0 && block_ && block_->ptr) {
      std::memcpy(cpu->get_storage().data(), block_->ptr, size_ * sizeof(T));
    }
    return cpu;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> clone() const override
  {
    auto copy = std::make_unique<IntelBufferImpl<T>>(size_);
    if (size_ > 0 && block_ && block_->ptr && copy->block_ && copy->block_->ptr) {
      std::memcpy(copy->block_->ptr, block_->ptr, size_ * sizeof(T));
    }
    return copy;
  }


  T * data() { return block_ ? static_cast<T *>(block_->ptr) : nullptr; }
  const T * data() const { return block_ ? static_cast<const T *>(block_->ptr) : nullptr; }

  /// Get the underlying pool block (for descriptor creation).
  PoolBlock * get_block() { return block_; }
  const PoolBlock * get_block() const { return block_; }

  /// Access the global memory pool singleton.
  static std::shared_ptr<IntelMemoryBufferPool> get_or_create_global_pool()
  {
    static std::shared_ptr<IntelMemoryBufferPool> global_pool = [] {
        auto pool = std::make_shared<IntelMemoryBufferPool>();
        if (!pool->initialize()) {
          return std::shared_ptr<IntelMemoryBufferPool>(nullptr);
        }
        return pool;
      }();
    return global_pool;
  }

  /// Check if the pool supports IPC (either Level Zero or DMA-BUF).
  static bool is_ipc_capable()
  {
    auto pool = get_or_create_global_pool();
    return pool && (pool->is_level_zero_capable() || pool->is_dmabuf_capable());
  }

private:
  void allocate(size_t n)
  {
    auto pool = get_or_create_global_pool();
    if (!pool) {
      return;
    }
    block_ = pool->allocate(n * sizeof(T));
    if (block_) {
      owns_block_ = true;
    }
  }

  static void free_standalone_resource(PoolBlock * block)
  {
    if (block->transport == intel_memory_core::TransportMode::DMA_BUF) {
      if (block->ptr && block->ptr != MAP_FAILED) {
        munmap(block->ptr, block->size);
      }
      if (block->dmabuf_fd >= 0) {
        close(block->dmabuf_fd);
      }
    }
  }

  size_t size_;
  PoolBlock * block_;
  bool owns_block_{false};
  bool standalone_{false};
  bool owns_mapping_{false};
};

}  // namespace intel_buffer_backend

#endif  // INTEL_BUFFER__INTEL_BUFFER_IMPL_HPP_
