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

#include <memory>

#include <gtest/gtest.h>

#include "intel_memory_core/intel_memory_buffer_pool.hpp"

using intel_memory_core::IntelMemoryBufferPool;
using intel_memory_core::PoolConfig;
using intel_memory_core::TransportMode;

namespace
{
std::shared_ptr<IntelMemoryBufferPool> make_initialized_pool(
  const PoolConfig & config = PoolConfig{})
{
  auto pool = std::make_shared<IntelMemoryBufferPool>();
  EXPECT_TRUE(pool->initialize(config));
  return pool;
}
}  // namespace

TEST(IntelMemoryBufferPool, InitializeIsConsistentWithProbedCapabilities)
{
  auto pool = make_initialized_pool();
  if (pool->is_level_zero_capable()) {
    EXPECT_EQ(pool->transport_mode(), TransportMode::LEVEL_ZERO_USM);
  } else if (pool->is_dmabuf_capable()) {
    EXPECT_EQ(pool->transport_mode(), TransportMode::DMA_BUF);
  } else {
    GTEST_SKIP() << "Neither Level Zero nor DMA-BUF available on this "
                    "machine; CPU-fallback mode exercised below instead";
  }
}

TEST(IntelMemoryBufferPool, NeitherTransportStillInitializesInCpuFallbackMode)
{
  auto pool = std::make_shared<IntelMemoryBufferPool>();
  bool ok = pool->initialize();
  ASSERT_TRUE(ok);
  if (pool->is_level_zero_capable() || pool->is_dmabuf_capable()) {
    GTEST_SKIP() << "This machine has a capable transport; the CPU-fallback "
                    "branch (neither L0 nor DMA-BUF) cannot be exercised here";
  }
  EXPECT_TRUE(ok);
}

TEST(IntelMemoryBufferPool, AllocateReturnsAlignedNonNullBlockWhenCapable)
{
  auto pool = make_initialized_pool();
  if (!pool->is_level_zero_capable() && !pool->is_dmabuf_capable()) {
    GTEST_SKIP() << "No capable transport on this machine";
  }
  auto * block = pool->allocate(100);
  ASSERT_NE(block, nullptr);
  EXPECT_GE(block->size, 100u);
  EXPECT_EQ(block->size % 65536u, 0u);
  EXPECT_NE(block->ptr, nullptr);
  pool->release(block);
}

TEST(IntelMemoryBufferPool, AssignUidIsStableAndUniquePerBlock)
{
  auto pool = make_initialized_pool();
  if (!pool->is_level_zero_capable() && !pool->is_dmabuf_capable()) {
    GTEST_SKIP() << "No capable transport on this machine";
  }
  auto * block_a = pool->allocate(4096);
  auto * block_b = pool->allocate(4096);
  ASSERT_NE(block_a, nullptr);
  ASSERT_NE(block_b, nullptr);
  ASSERT_NE(block_a, block_b);

  uint64_t uid_a = pool->assign_uid(block_a);
  uint64_t uid_b = pool->assign_uid(block_b);
  EXPECT_NE(uid_a, 0u);
  EXPECT_NE(uid_b, 0u);
  EXPECT_NE(uid_a, uid_b);
  // Re-assigning to the same block returns the same uid.
  EXPECT_EQ(pool->assign_uid(block_a), uid_a);

  pool->release(block_a);
  pool->release(block_b);
}

TEST(IntelMemoryBufferPool, ReleaseThenAllocateReusesTheSameBlock)
{
  auto pool = make_initialized_pool();
  if (!pool->is_level_zero_capable() && !pool->is_dmabuf_capable()) {
    GTEST_SKIP() << "No capable transport on this machine";
  }
  auto * block1 = pool->allocate(8192);
  ASSERT_NE(block1, nullptr);
  size_t total_before = pool->total_blocks();

  pool->release(block1);
  EXPECT_EQ(pool->free_blocks(), 1u);

  auto * block2 = pool->allocate(8192);
  ASSERT_NE(block2, nullptr);
  EXPECT_EQ(block2, block1);
  EXPECT_EQ(pool->total_blocks(), total_before);
  EXPECT_EQ(pool->free_blocks(), 0u);

  pool->release(block2);
}

TEST(IntelMemoryBufferPool, MaxBlocksExhaustionReturnsNullptr)
{
  PoolConfig config;
  config.max_blocks = 1;
  auto pool = make_initialized_pool(config);
  if (!pool->is_level_zero_capable() && !pool->is_dmabuf_capable()) {
    GTEST_SKIP() << "No capable transport on this machine";
  }
  auto * block1 = pool->allocate(4096);
  ASSERT_NE(block1, nullptr);

  auto * block2 = pool->allocate(4096);
  EXPECT_EQ(block2, nullptr);

  pool->release(block1);
}

TEST(IntelMemoryBufferPool, FindBlockForPtrLocatesOwningBlockOnly)
{
  auto pool = make_initialized_pool();
  if (!pool->is_level_zero_capable() && !pool->is_dmabuf_capable()) {
    GTEST_SKIP() << "No capable transport on this machine";
  }
  auto * block = pool->allocate(4096);
  ASSERT_NE(block, nullptr);

  EXPECT_EQ(pool->find_block_for_ptr(block->ptr), block);
  int unrelated_stack_value = 0;
  EXPECT_EQ(pool->find_block_for_ptr(&unrelated_stack_value), nullptr);

  pool->release(block);
}

TEST(IntelMemoryBufferPool, ActiveAndFreeBlocksTrackTotal)
{
  auto pool = make_initialized_pool();
  if (!pool->is_level_zero_capable() && !pool->is_dmabuf_capable()) {
    GTEST_SKIP() << "No capable transport on this machine";
  }
  auto * block = pool->allocate(4096);
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(pool->active_blocks() + pool->free_blocks(), pool->total_blocks());
  EXPECT_EQ(pool->active_blocks(), pool->total_blocks());

  pool->release(block);
  EXPECT_EQ(pool->active_blocks(), 0u);
  EXPECT_EQ(pool->free_blocks(), pool->total_blocks());
}
