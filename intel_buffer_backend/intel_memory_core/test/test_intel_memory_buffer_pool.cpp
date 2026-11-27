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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <atomic>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/intel_memory_buffer_pool.hpp"

using intel_memory_core::IntelMemoryBufferPool;
using intel_memory_core::PoolBlock;
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

// --- Reuse must never hand out a block a consumer is still reading -----------

namespace
{
// What a consumer import records on a block: a reference plus its pid.
void hold_as_consumer(intel_memory_core::PoolBlock * block)
{
  block->ipc_meta->refcount.fetch_add(1);
  block->ipc_meta->holder_pids[0].store(static_cast<int32_t>(getpid()));
}

void release_as_consumer(intel_memory_core::PoolBlock * block)
{
  block->ipc_meta->holder_pids[0].store(0);
  block->ipc_meta->refcount.store(0);
}
}  // namespace

TEST(IntelMemoryBufferPoolReuse, ExhaustedPoolNeverHandsOutABlockAConsumerHolds)
{
  PoolConfig config;
  config.max_blocks = 2;
  auto pool = make_initialized_pool(config);
  if (!pool->is_level_zero_capable() && !pool->is_dmabuf_capable()) {
    GTEST_SKIP() << "No capable transport on this machine";
  }
  auto * a = pool->allocate(100);
  auto * b = pool->allocate(100);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  if (!a->ipc_meta || !b->ipc_meta) {
    GTEST_SKIP() << "pool blocks have no IPC metadata on this machine";
  }
  hold_as_consumer(a);
  hold_as_consumer(b);
  pool->release(a);  // producer is done with both; consumers still read them
  pool->release(b);

  auto * c = pool->allocate(100);
  EXPECT_EQ(c, nullptr) << "pool handed out a block a live consumer still holds";
  EXPECT_EQ(a->ipc_meta->refcount.load(), 1);
  EXPECT_EQ(b->ipc_meta->refcount.load(), 1);

  release_as_consumer(a);
  release_as_consumer(b);
  if (c != nullptr) {
    pool->release(c);
  }
}

TEST(IntelMemoryBufferPoolReuse, ReusedBlockInvalidatesItsPublishedUid)
{
  // Reclaimed early (exhausted pool, grace period not over) and normally
  // (grace period over): either way, descriptors of the frame the block held
  // before must stop validating once the producer can write into it again.
  for (bool after_grace : {false, true}) {
    PoolConfig config;
    config.max_blocks = 1;
    config.grace_period_us = after_grace ? 1000 : 60 * 1000 * 1000;
    auto pool = make_initialized_pool(config);
    if (!pool->is_level_zero_capable() && !pool->is_dmabuf_capable()) {
      GTEST_SKIP() << "No capable transport on this machine";
    }
    auto * block = pool->allocate(100);
    ASSERT_NE(block, nullptr);
    if (!block->ipc_meta) {
      GTEST_SKIP() << "pool blocks have no IPC metadata on this machine";
    }
    const uint64_t published_uid = pool->assign_uid(block);
    pool->release(block);
    if (after_grace) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto * reused = pool->allocate(100);
    ASSERT_EQ(reused, block);
    EXPECT_NE(reused->ipc_meta->uid.load(), published_uid)
      << (after_grace ? "normal" : "exhausted-pool")
      << " reuse left the previous frame's uid valid";
    pool->release(reused);
  }
}

// --- Application-owned dma-bufs (wrap_external_dmabuf) ------------------------

namespace
{
int make_memfd(size_t size)
{
  const int fd = memfd_create("imc_external_dmabuf", 0);
  if (fd >= 0 && ftruncate(fd, static_cast<off_t>(size)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}
}  // namespace

TEST(IntelMemoryBufferPool, ExternalDmabufGetsItsOwnBlockIdAndRefcountMetadata)
{
  auto pool = make_initialized_pool();
  PoolBlock * pool_block = pool->allocate(4096);
  const int fd = make_memfd(4096);
  ASSERT_GE(fd, 0);

  PoolBlock * external = pool->wrap_external_dmabuf(fd, 4096);
  ASSERT_NE(external, nullptr);
  EXPECT_TRUE(external->external);
  EXPECT_NE(external->ipc_meta, nullptr);
  if (pool_block) {
    EXPECT_NE(external->block_id, pool_block->block_id);
  }
  // Same buffer, same block; one producer reference per wrap.
  EXPECT_EQ(pool->wrap_external_dmabuf(fd, 4096), external);
  EXPECT_EQ(external->producer_refs, 2u);
  EXPECT_EQ(pool->wrap_external_dmabuf(fd, 8192), nullptr);   // larger than the buffer
  pool->release(external);
  pool->release(external);
  EXPECT_EQ(external->producer_refs, 0u);
  close(fd);
}

TEST(IntelMemoryBufferPool, ExternalDmabufEachWrapPublishesUnderANewUid)
{
  auto pool = make_initialized_pool();
  const int fd = make_memfd(4096);
  ASSERT_GE(fd, 0);

  PoolBlock * block = pool->wrap_external_dmabuf(fd, 4096);
  ASSERT_NE(block, nullptr);
  const uint64_t first = pool->assign_uid(block);
  pool->release(block);
  ASSERT_TRUE(pool->try_reclaim_external_dmabuf(fd));
  ASSERT_EQ(pool->wrap_external_dmabuf(fd, 4096), block);
  EXPECT_NE(pool->assign_uid(block), first);
  pool->release(block);
  close(fd);
}

TEST(IntelMemoryBufferPool, ExternalDmabufNeverImportedDescriptorIsGivenUpAfterGracePeriod)
{
  PoolConfig config;
  config.grace_period_us = 2000;
  auto pool = make_initialized_pool(config);
  const int fd = make_memfd(4096);
  ASSERT_GE(fd, 0);

  PoolBlock * block = pool->wrap_external_dmabuf(fd, 4096);
  ASSERT_NE(block, nullptr);
  ASSERT_NE(block->ipc_meta, nullptr);
  pool->assign_uid(block);
  block->ipc_meta->pending_imports.store(1);   // descriptor issued, never imported
  pool->release(block);

  EXPECT_FALSE(pool->try_reclaim_external_dmabuf(fd));
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_TRUE(pool->try_reclaim_external_dmabuf(fd));
  EXPECT_EQ(block->ipc_meta->pending_imports.load(), 0);
  close(fd);
}

TEST(IntelMemoryBufferPool, ExternalDmabufRegistryIsBoundedAndEvictsOnlyUnusedBuffers)
{
  PoolConfig config;
  config.max_blocks = 1;
  auto pool = make_initialized_pool(config);
  const int fd_a = make_memfd(4096);
  const int fd_b = make_memfd(4096);
  ASSERT_GE(fd_a, 0);
  ASSERT_GE(fd_b, 0);

  PoolBlock * a = pool->wrap_external_dmabuf(fd_a, 4096);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(pool->wrap_external_dmabuf(fd_b, 4096), nullptr);   // a is still wrapped
  pool->release(a);
  PoolBlock * b = pool->wrap_external_dmabuf(fd_b, 4096);
  EXPECT_NE(b, nullptr);   // a was evicted
  if (b) {
    pool->release(b);
  }
  close(fd_a);
  close(fd_b);
}

// Every publisher thread shares the process-wide pool. Concurrent publishes of
// one block agree on a single uid (the one subscribers validate against), and
// of distinct blocks never share one.
TEST(IntelMemoryBufferPool, ConcurrentAssignUidAgreesPerBlockAndIsUniqueAcrossBlocks)
{
  auto pool = std::make_shared<IntelMemoryBufferPool>();
  ASSERT_TRUE(pool->initialize());
  constexpr int kThreads = 8;
  constexpr int kRounds = 200;

  std::vector<PoolBlock *> blocks;
  for (int i = 0; i < kThreads; ++i) {
    PoolBlock * block = pool->allocate(4096);
    if (!block || !block->ipc_meta) {
      GTEST_SKIP() << "no pool block with refcount metadata on this machine";
    }
    blocks.push_back(block);
  }
  PoolBlock * shared = blocks[0];

  for (int round = 0; round < kRounds; ++round) {
    for (PoolBlock * block : blocks) {
      block->current_uid = 0;   // a new frame in every block
    }
    std::atomic<bool> go{false};
    std::vector<uint64_t> shared_uids(kThreads), own_uids(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
          while (!go.load()) {}
          shared_uids[t] = pool->assign_uid(shared);
          own_uids[t] = t == 0 ? shared_uids[t] : pool->assign_uid(blocks[t]);
        });
    }
    go.store(true);
    for (auto & thread : threads) {
      thread.join();
    }
    for (int t = 0; t < kThreads; ++t) {
      ASSERT_EQ(shared_uids[t], shared->ipc_meta->uid.load()) << "round " << round;
    }
    EXPECT_EQ(std::set<uint64_t>(own_uids.begin(), own_uids.end()).size(), size_t{kThreads});
  }
  for (PoolBlock * block : blocks) {
    pool->release(block);
  }
}

// Two containers sharing /dev/shm (e.g. --ipc=host) have separate PID
// namespaces, so another producer can run under this process's pid. Creating
// this pool's metadata must never replace that producer's segment.
TEST(IntelMemoryBufferPool, MetadataNeverReplacesAnotherProducersSegmentWithTheSamePid)
{
  auto pool = std::make_shared<IntelMemoryBufferPool>();
  ASSERT_TRUE(pool->initialize());
  // The other producer's metadata for its block 0, named the way it would be
  // without this pool's id.
  const std::string other = "/intel_memory_buffer_" + std::to_string(getpid()) + "_0";
  const int fd = shm_open(other.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  ASSERT_GE(fd, 0) << "stale " << other << " in /dev/shm";
  ASSERT_EQ(ftruncate(fd, sizeof(intel_memory_core::IPCMetadata)), 0);
  auto * theirs = static_cast<intel_memory_core::IPCMetadata *>(mmap(nullptr,
    sizeof(intel_memory_core::IPCMetadata), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  ASSERT_NE(theirs, MAP_FAILED);
  theirs->uid.store(0xc0ffee);
  struct stat before{};
  ASSERT_EQ(fstat(fd, &before), 0);

  PoolBlock * block = pool->allocate(4096);
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->block_id, 0u);

  const int again = shm_open(other.c_str(), O_RDWR, 0);
  struct stat after{};
  EXPECT_TRUE(again >= 0 && fstat(again, &after) == 0 && after.st_ino == before.st_ino) <<
    "the other producer's segment was unlinked and replaced";
  EXPECT_EQ(theirs->uid.load(), 0xc0ffeeu);
  EXPECT_NE(block->ipc_meta, theirs);
  if (again >= 0) {close(again);}
  pool->release(block);
  munmap(theirs, sizeof(intel_memory_core::IPCMetadata));
  close(fd);
  shm_unlink(other.c_str());
}

// Even a segment at this pool's own name (left by another process) is never
// replaced: the block then has no metadata and is sent as a copy.
TEST(IntelMemoryBufferPool, MetadataIsNotCreatedOverAnExistingSegment)
{
  auto pool = std::make_shared<IntelMemoryBufferPool>();
  ASSERT_TRUE(pool->initialize());
  const std::string name = intel_memory_core::ipc_meta_shm_name(getpid(), pool->pool_id(), 0);
  const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  ASSERT_GE(fd, 0);
  struct stat before{};
  ASSERT_EQ(fstat(fd, &before), 0);

  PoolBlock * block = pool->allocate(4096);
  ASSERT_NE(block, nullptr);
  ASSERT_EQ(block->block_id, 0u);
  EXPECT_EQ(block->ipc_meta, nullptr);
  struct stat after{};
  const int again = shm_open(name.c_str(), O_RDWR, 0);
  EXPECT_TRUE(again >= 0 && fstat(again, &after) == 0 && after.st_ino == before.st_ino);
  if (again >= 0) {close(again);}
  pool->release(block);
  close(fd);
  shm_unlink(name.c_str());
}
