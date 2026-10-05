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

#include <cstdlib>
#include <cstring>
#include <memory>

#include <gtest/gtest.h>

#include "intel_buffer/intel_buffer_impl.hpp"

using intel_buffer_backend::IntelBufferImpl;
using intel_buffer_backend::PoolBlock;

namespace
{

class AdoptedHeapBlock
{
public:
  explicit AdoptedHeapBlock(size_t size)
  : size_(size), buffer_(static_cast<uint8_t *>(std::malloc(size)))
  {
    block_ = new PoolBlock();
    block_->ptr = buffer_;
    block_->size = size;
  }
  ~AdoptedHeapBlock()
  {
    delete block_;
    std::free(buffer_);
  }
  PoolBlock * block() {return block_;}
  uint8_t * buffer() {return buffer_;}
  size_t size() const {return size_;}

private:
  size_t size_;
  uint8_t * buffer_;
  PoolBlock * block_;
};
}  // namespace

TEST(IntelBufferImpl, DefaultConstructedIsEmpty)
{
  IntelBufferImpl<uint8_t> impl;
  EXPECT_EQ(impl.size(), 0u);
  EXPECT_EQ(impl.data(), nullptr);
  EXPECT_EQ(impl.get_block(), nullptr);
}

TEST(IntelBufferImpl, AdoptCtorOverHeapBlockDoesNotOwnIt)
{
  AdoptedHeapBlock heap(64);
  std::memset(heap.buffer(), 0x11, heap.size());

  IntelBufferImpl<uint8_t> impl(heap.block(), heap.size(), false);
  EXPECT_EQ(impl.size(), heap.size());
  EXPECT_EQ(impl.data(), heap.buffer());
  EXPECT_EQ(impl.get_block(), heap.block());
}

TEST(IntelBufferImpl, GetBackendTypeIsIntelBuffer)
{
  IntelBufferImpl<uint8_t> impl;
  EXPECT_EQ(impl.get_backend_type(), "intel_buffer");
}

TEST(IntelBufferImpl, ToCpuOverAdoptedBlockCopiesBytes)
{
  AdoptedHeapBlock heap(128);
  for (size_t i = 0; i < heap.size(); ++i) {
    heap.buffer()[i] = static_cast<uint8_t>(i);
  }

  IntelBufferImpl<uint8_t> impl(heap.block(), heap.size(), false);
  auto cpu = impl.to_cpu();
  ASSERT_NE(cpu, nullptr);
  ASSERT_EQ(cpu->size(), heap.size());
  auto * cpu_typed = dynamic_cast<rosidl::CpuBufferImpl<uint8_t> *>(cpu.get());
  ASSERT_NE(cpu_typed, nullptr);
  EXPECT_EQ(std::memcmp(cpu_typed->get_storage().data(), heap.buffer(), heap.size()), 0);
}

TEST(IntelBufferImpl, ToCpuOnEmptyImplGuardsCleanly)
{
  IntelBufferImpl<uint8_t> impl;
  auto cpu = impl.to_cpu();
  ASSERT_NE(cpu, nullptr);
  EXPECT_EQ(cpu->size(), 0u);
}

TEST(IntelBufferImpl, CloneOnEmptyImplGuardsCleanly)
{
  IntelBufferImpl<uint8_t> impl;
  auto copy = impl.clone();
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->size(), 0u);
}

TEST(IntelBufferImpl, CloneProducesIndependentCopyWhenPoolAvailable)
{
  if (!IntelBufferImpl<uint8_t>::get_or_create_global_pool()) {
    GTEST_SKIP() << "No USM/DMA-BUF pool available on this machine; clone() "
                    "allocates through the global pool, which is required "
                    "for this assertion";
  }
  AdoptedHeapBlock heap(64);
  std::memset(heap.buffer(), 0x77, heap.size());

  IntelBufferImpl<uint8_t> impl(heap.block(), heap.size(), false);
  auto copy = impl.clone();
  ASSERT_NE(copy, nullptr);
  ASSERT_EQ(copy->size(), heap.size());

  auto * copy_typed = dynamic_cast<IntelBufferImpl<uint8_t> *>(copy.get());
  ASSERT_NE(copy_typed, nullptr);
  ASSERT_NE(copy_typed->data(), nullptr);
  EXPECT_NE(copy_typed->data(), impl.data());
  EXPECT_EQ(std::memcmp(copy_typed->data(), heap.buffer(), heap.size()), 0);
}

TEST(IntelBufferImpl, GetOrCreateGlobalPoolIsProcessWideSingleton)
{
  auto pool_a = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
  auto pool_b = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
  EXPECT_EQ(pool_a, pool_b);
}

TEST(IntelBufferImpl, IsIpcCapableMatchesPoolCapabilities)
{
  auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
  bool expected = pool && (pool->is_level_zero_capable() || pool->is_dmabuf_capable());
  EXPECT_EQ(IntelBufferImpl<uint8_t>::is_ipc_capable(), expected);
}
