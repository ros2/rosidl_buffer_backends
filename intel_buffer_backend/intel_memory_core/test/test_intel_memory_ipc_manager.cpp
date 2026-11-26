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
#include <unistd.h>

#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/intel_memory_buffer_pool.hpp"
#include "intel_memory_core/intel_memory_ipc_manager.hpp"
#include "intel_memory_core/l0_usm_ipc.hpp"

using intel_memory_core::DmaBufIpc;
using intel_memory_core::IntelMemoryBufferPool;
using intel_memory_core::IntelMemoryIPCManager;
using intel_memory_core::IpcImportResult;
using intel_memory_core::L0UsmIpc;
using intel_memory_core::PoolBlock;
using intel_memory_core::TransportMode;

namespace
{
// A descriptor size no Level Zero allocation in these tests reaches.
constexpr uint64_t kBeyondAnyMapping = uint64_t{1} << 40;

struct ExportedL0Block
{
  PoolBlock * block{nullptr};
  uint8_t handle[64] = {};
  uint64_t handle_size{0};
};

// A Level Zero USM pool block with its IPC handle exported, or nullptr block.
ExportedL0Block export_l0_block(IntelMemoryBufferPool & pool)
{
  ExportedL0Block exported;
  PoolBlock * block = pool.allocate(4096);
  uint64_t context_id = 0;
  if (block && L0UsmIpc::export_handle(block, exported.handle, exported.handle_size, context_id)) {
    exported.block = block;
  }
  return exported;
}

IpcImportResult import_l0(
  const IntelMemoryBufferPool & pool, const ExportedL0Block & exported, uint64_t size,
  const std::string & socket_path)
{
  return L0UsmIpc::import_handle(
    exported.handle, exported.handle_size, exported.block->ze_device_ordinal,
    exported.block->ze_device_uuid, size, static_cast<int32_t>(getpid()),
    exported.block->block_id, socket_path, pool.pool_id());
}

std::shared_ptr<IntelMemoryBufferPool> make_l0_pool()
{
  auto pool = std::make_shared<IntelMemoryBufferPool>();
  if (!pool->initialize() || !pool->is_level_zero_capable() ||
    pool->transport_mode() != TransportMode::LEVEL_ZERO_USM)
  {
    return nullptr;
  }
  return pool;
}
}  // namespace

// Descriptor sizes are untrusted: an import never returns a pointer with fewer
// mapped bytes behind it than the descriptor claims, also on a cache hit.
TEST(L0UsmIpcImport, DmabufFdImportMapsWholeBlockAndRejectsLargerSizes)
{
  auto pool = make_l0_pool();
  if (!pool) {
    GTEST_SKIP() << "no Level Zero USM pool on this machine";
  }
  ExportedL0Block exported = export_l0_block(*pool);
  if (!exported.block) {
    GTEST_SKIP() << "pool block not exportable on this machine";
  }
  const std::string socket_path = pool->register_block_for_ipc(exported.block);
  if (socket_path.empty()) {
    GTEST_SKIP() << "no dma-buf fd export for Level Zero blocks on this machine";
  }

  // The first frame of a block can be smaller than later ones.
  IpcImportResult first = import_l0(*pool, exported, 1, socket_path);
  if (!first.ptr) {
    GTEST_SKIP() << "Level Zero dma-buf fd import unavailable on this machine";
  }
  IpcImportResult whole = import_l0(*pool, exported, exported.block->size, socket_path);
  EXPECT_EQ(whole.ptr, first.ptr);
  EXPECT_EQ(import_l0(*pool, exported, kBeyondAnyMapping, socket_path).ptr, nullptr);
  EXPECT_EQ(import_l0(*pool, exported, 0, socket_path).ptr, nullptr);   // as a fresh import
}

// Evicting releases both kinds of cached mapping without an error, and a closed
// handle can be opened again. (Drivers may accept zeMemFree() on a mapping from
// zeMemOpenIpcHandle(); the mismatch then only shows in the Level Zero leak
// checker: ZE_ENABLE_VALIDATION_LAYER=1 ZEL_ENABLE_BASIC_LEAK_CHECKER=1.)
TEST(L0UsmIpcImport, EvictReleasesBothImportKindsAndHandleCanBeReopened)
{
  auto pool = make_l0_pool();
  if (!pool) {
    GTEST_SKIP() << "no Level Zero USM pool on this machine";
  }
  ExportedL0Block by_handle = export_l0_block(*pool);
  ExportedL0Block by_fd = export_l0_block(*pool);
  if (!by_handle.block || !by_fd.block) {
    GTEST_SKIP() << "pool block not exportable on this machine";
  }
  const std::string socket_path = pool->register_block_for_ipc(by_fd.block);
  if (!import_l0(*pool, by_handle, by_handle.block->size, "").ptr ||
    socket_path.empty() || !import_l0(*pool, by_fd, by_fd.block->size, socket_path).ptr)
  {
    GTEST_SKIP() << "Level Zero imports unavailable in-process on this machine";
  }

  testing::internal::CaptureStderr();
  L0UsmIpc::evict_pool(pool->pool_id());
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(log.find("releasing an imported mapping"), std::string::npos) << log;

  // The handle was closed, so it can be opened again.
  EXPECT_NE(import_l0(*pool, by_handle, by_handle.block->size, "").ptr, nullptr);
}

TEST(L0UsmIpcImport, IpcHandleImportRejectsSizeBeyondAllocation)
{
  auto pool = make_l0_pool();
  if (!pool) {
    GTEST_SKIP() << "no Level Zero USM pool on this machine";
  }
  ExportedL0Block fits = export_l0_block(*pool);
  ExportedL0Block too_large = export_l0_block(*pool);
  if (!fits.block || !too_large.block) {
    GTEST_SKIP() << "pool block not exportable on this machine";
  }

  IpcImportResult imported = import_l0(*pool, fits, fits.block->size, "");
  if (!imported.ptr) {
    GTEST_SKIP() << "zeMemOpenIpcHandle unavailable in-process on this machine";
  }
  // Cache hit, then a fresh import: both are checked against the allocation.
  EXPECT_EQ(import_l0(*pool, fits, kBeyondAnyMapping, "").ptr, nullptr);
  EXPECT_EQ(import_l0(*pool, fits, 0, "").ptr, nullptr);
  EXPECT_EQ(import_l0(*pool, too_large, kBeyondAnyMapping, "").ptr, nullptr);
}

// Messages can outlive the thread that took them (a user queue, an executor
// whose threads are joined when spin() returns): its Level Zero imports must
// stay mapped after that thread exits. Both import kinds: IPC handle and fd.
TEST(L0UsmIpcImport, ImportOutlivesTheImportingThread)
{
  auto pool = make_l0_pool();
  if (!pool) {
    GTEST_SKIP() << "no Level Zero USM pool on this machine";
  }
  ExportedL0Block by_handle = export_l0_block(*pool);
  ExportedL0Block by_fd = export_l0_block(*pool);
  if (!by_handle.block || !by_fd.block) {
    GTEST_SKIP() << "pool block not exportable on this machine";
  }
  const std::string socket_path = pool->register_block_for_ipc(by_fd.block);

  for (const auto & [exported, path] : {std::pair{by_handle, std::string()},
      std::pair{by_fd, socket_path}})
  {
    if (exported.block == by_fd.block && path.empty()) {
      continue;   // no dma-buf fd export for Level Zero blocks on this machine
    }
    auto * producer = static_cast<uint8_t *>(exported.block->ptr);
    producer[0] = 0x5a;
    producer[exported.block->size - 1] = 0xa5;

    IpcImportResult imported;
    std::thread([&] {imported = import_l0(*pool, exported, exported.block->size, path);}).join();
    if (!imported.ptr) {
      GTEST_SKIP() << "Level Zero imports unavailable in-process on this machine";
    }
    auto * consumer = static_cast<volatile uint8_t *>(imported.ptr);
    EXPECT_EQ(consumer[0], 0x5a) << (path.empty() ? "IPC handle" : "fd");
    EXPECT_EQ(consumer[exported.block->size - 1], 0xa5) << (path.empty() ? "IPC handle" : "fd");
  }
}

// Importing threads share one context, and each keeps its own import cache:
// one thread releasing its import leaves another thread's import of the same
// block mapped.
TEST(L0UsmIpcImport, ReleasingOneThreadsImportKeepsAnotherThreadsImport)
{
  auto pool = make_l0_pool();
  if (!pool) {
    GTEST_SKIP() << "no Level Zero USM pool on this machine";
  }
  ExportedL0Block by_handle = export_l0_block(*pool);
  ExportedL0Block by_fd = export_l0_block(*pool);
  if (!by_handle.block || !by_fd.block) {
    GTEST_SKIP() << "pool block not exportable on this machine";
  }
  const std::string socket_path = pool->register_block_for_ipc(by_fd.block);

  for (const auto & [exported, path] : {std::pair{by_handle, std::string()},
      std::pair{by_fd, socket_path}})
  {
    if (exported.block == by_fd.block && path.empty()) {
      continue;
    }
    static_cast<uint8_t *>(exported.block->ptr)[0] = 0x3c;

    IpcImportResult kept;
    IpcImportResult released;
    std::thread([&] {kept = import_l0(*pool, exported, exported.block->size, path);}).join();
    std::thread([&] {
        released = import_l0(*pool, exported, exported.block->size, path);
        L0UsmIpc::evict_pool(pool->pool_id());
      }).join();
    if (!kept.ptr || !released.ptr) {
      GTEST_SKIP() << "Level Zero imports unavailable in-process on this machine";
    }
    EXPECT_EQ(static_cast<volatile uint8_t *>(kept.ptr)[0], 0x3c) <<
      (path.empty() ? "IPC handle" : "fd");
  }
}

TEST(IntelMemoryIpcManager, ImportDmabufBlockWithBogusInputsNeverThrows)
{
  IntelMemoryIPCManager::ImportResult result;
  ASSERT_NO_THROW(
    result = IntelMemoryIPCManager::import_dmabuf_block(
      "/tmp/does_not_exist.sock", 999999, 42, 4096, 0, 1));
  EXPECT_EQ(result.ptr, nullptr);
  EXPECT_EQ(result.dmabuf_fd, -1);
}

TEST(IntelMemoryIpcManager, ImportLevelZeroBlockWithBogusInputsNeverThrows)
{
  uint8_t fake_handle[8] = {};
  uint8_t fake_uuid[16] = {};
  IntelMemoryIPCManager::ImportResult result;
  ASSERT_NO_THROW(
    result = IntelMemoryIPCManager::import_level_zero_block(
      fake_handle, sizeof(fake_handle),
      -1, fake_uuid, 4096,
      999999, 42, 0,
      "/tmp/does_not_exist_l0.sock", 1));
  EXPECT_EQ(result.ptr, nullptr);
}

TEST(IntelMemoryIpcManager, ImportLevelZeroBlockWithNullUuidFallsBackToOrdinal)
{
  uint8_t fake_handle[8] = {};
  IntelMemoryIPCManager::ImportResult result;
  ASSERT_NO_THROW(
    result = IntelMemoryIPCManager::import_level_zero_block(
      fake_handle, sizeof(fake_handle),
      -1, nullptr, 4096,
      999999, 42, 0,
      "", 1));
  EXPECT_EQ(result.ptr, nullptr);
}

TEST(IntelMemoryIpcManager, ReleaseBlockWithNullMetaIsNoOp)
{
  EXPECT_NO_THROW(IntelMemoryIPCManager::release_block(nullptr));
}

TEST(IntelMemoryIpcManager, RegisterFdDelegatesToDmaBufIpc)
{
  std::string path;
  ASSERT_NO_THROW(path = IntelMemoryIPCManager::register_fd(0xabcdu, -1));
  EXPECT_FALSE(path.empty());
  DmaBufIpc::unregister_block(0xabcdu);
}

// --- Imports must hold a validated reference ----------------------------------

TEST(IntelMemoryIpcManager, ImportWithUidZeroIsRejected)
{
  auto pool = std::make_shared<IntelMemoryBufferPool>();
  ASSERT_TRUE(pool->initialize());
  PoolBlock * block = pool->allocate(4096);
  if (!block || !block->ipc_meta) {
    GTEST_SKIP() << "no pool block with refcount metadata on this machine";
  }
  const uint64_t uid = pool->assign_uid(block);

  EXPECT_EQ(IntelMemoryIPCManager::acquire_local_block(pool->pool_id(), block->block_id, 0),
    nullptr);
  EXPECT_EQ(block->ipc_meta->refcount.load(), 0);

  auto * meta = IntelMemoryIPCManager::acquire_local_block(pool->pool_id(), block->block_id, uid);
  ASSERT_NE(meta, nullptr);
  IntelMemoryIPCManager::release_block(meta);
  pool->release(block);
}

TEST(IntelMemoryIpcManager, ImportWithoutRefcountMetadataIsRejected)
{
  // An exported buffer whose producer has no metadata segment for the block.
  const uint32_t block_id = 0xbeef0001u;
  const int fd = memfd_create("imc_no_meta", 0);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, 4096), 0);
  const std::string socket_path = DmaBufIpc::register_fd(block_id, fd);
  ASSERT_FALSE(socket_path.empty());

  auto result = IntelMemoryIPCManager::import_dmabuf_block(
    socket_path, static_cast<int32_t>(getpid()), block_id, 4096, /*expected_uid=*/1, 0x1234u);
  EXPECT_EQ(result.ptr, nullptr);
  EXPECT_EQ(result.ipc_meta, nullptr);

  DmaBufIpc::unregister_block(block_id);
  close(fd);
}

namespace
{
// A dma-buf (a memfd stands in) of `block_id` exported over the real socket,
// with refcount metadata published under kUid.
struct ExportedDmabuf
{
  static constexpr uint64_t kUid = 0x5eed;
  static constexpr uint64_t kBytes = 4096;

  ExportedDmabuf(uint32_t pool_id, uint32_t block_id)
  : block_id(block_id),
    meta_name(intel_memory_core::ipc_meta_shm_name(getpid(), pool_id, block_id))
  {
    fd = memfd_create("imc_evict", 0);
    if (fd >= 0 && ftruncate(fd, kBytes) == 0) {
      socket_path = DmaBufIpc::register_fd(block_id, fd);
    }
    const int meta_fd = shm_open(meta_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (meta_fd >= 0 && ftruncate(meta_fd, sizeof(intel_memory_core::IPCMetadata)) == 0) {
      void * p = mmap(nullptr, sizeof(intel_memory_core::IPCMetadata),
          PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd, 0);
      if (p != MAP_FAILED) {
        meta = new (p) intel_memory_core::IPCMetadata();
        meta->uid.store(kUid);
      }
    }
    if (meta_fd >= 0) {close(meta_fd);}
  }
  ~ExportedDmabuf()
  {
    DmaBufIpc::unregister_block(block_id);
    if (fd >= 0) {close(fd);}
    if (meta) {munmap(meta, sizeof(intel_memory_core::IPCMetadata));}
    shm_unlink(meta_name.c_str());
  }

  uint32_t block_id;
  std::string meta_name;
  int fd{-1};
  std::string socket_path;
  intel_memory_core::IPCMetadata * meta{nullptr};
};
}  // namespace

// A failed import of a producer instance (e.g. it's gone) drops that pool's
// cached mappings no buffer of this process still holds; held ones stay.
TEST(IntelMemoryIpcManager, FailedImportEvictsThePoolsUnheldMappings)
{
  const uint32_t pool_id = 0xe71c7001u;
  const int32_t pid = static_cast<int32_t>(getpid());
  ExportedDmabuf unheld(pool_id, 0xe71c0001u);
  ExportedDmabuf held(pool_id, 0xe71c0002u);
  ASSERT_FALSE(unheld.socket_path.empty());
  ASSERT_FALSE(held.socket_path.empty());
  ASSERT_NE(unheld.meta, nullptr);
  ASSERT_NE(held.meta, nullptr);

  auto first = IntelMemoryIPCManager::import_dmabuf_block(unheld.socket_path, pid,
    unheld.block_id, ExportedDmabuf::kBytes, ExportedDmabuf::kUid, pool_id);
  ASSERT_NE(first.ptr, nullptr);
  IntelMemoryIPCManager::release_block(first.ipc_meta);   // that buffer is gone
  auto kept = IntelMemoryIPCManager::import_dmabuf_block(held.socket_path, pid,
    held.block_id, ExportedDmabuf::kBytes, ExportedDmabuf::kUid, pool_id);
  ASSERT_NE(kept.ptr, nullptr);   // still referenced below

  // The producer's sockets are gone, and an import from it fails.
  DmaBufIpc::unregister_block(unheld.block_id);
  DmaBufIpc::unregister_block(held.block_id);
  EXPECT_EQ(IntelMemoryIPCManager::import_dmabuf_block("/tmp/imc_evict_gone.sock", pid,
    0xe71c0003u, ExportedDmabuf::kBytes, ExportedDmabuf::kUid, pool_id).ptr, nullptr);

  // The unheld mapping was dropped: importing it again needs its (gone) socket.
  EXPECT_EQ(IntelMemoryIPCManager::import_dmabuf_block(unheld.socket_path, pid,
    unheld.block_id, ExportedDmabuf::kBytes, ExportedDmabuf::kUid, pool_id).ptr, nullptr);
  // The held one is still mapped and served from the cache.
  auto again = IntelMemoryIPCManager::import_dmabuf_block(held.socket_path, pid,
    held.block_id, ExportedDmabuf::kBytes, ExportedDmabuf::kUid, pool_id);
  EXPECT_EQ(again.ptr, kept.ptr);
  IntelMemoryIPCManager::release_block(again.ipc_meta);
  IntelMemoryIPCManager::release_block(kept.ipc_meta);
}

// A cached mapping answers a descriptor with the same bounds as a fresh import:
// a zero size is rejected (and takes no reference) either way.
TEST(IntelMemoryIpcManager, ZeroSizeImportIsRejectedAlsoFromTheCache)
{
  const uint32_t pool_id = 0x5e20001u;
  const int32_t pid = static_cast<int32_t>(getpid());
  ExportedDmabuf exported(pool_id, 0x5e20002u);
  ASSERT_FALSE(exported.socket_path.empty());
  ASSERT_NE(exported.meta, nullptr);

  EXPECT_EQ(IntelMemoryIPCManager::import_dmabuf_block(exported.socket_path, pid,
    exported.block_id, 0, ExportedDmabuf::kUid, pool_id).ptr, nullptr);   // fresh
  auto whole = IntelMemoryIPCManager::import_dmabuf_block(exported.socket_path, pid,
    exported.block_id, ExportedDmabuf::kBytes, ExportedDmabuf::kUid, pool_id);
  ASSERT_NE(whole.ptr, nullptr);   // now cached, and held
  EXPECT_EQ(IntelMemoryIPCManager::import_dmabuf_block(exported.socket_path, pid,
    exported.block_id, 0, ExportedDmabuf::kUid, pool_id).ptr, nullptr);   // cache hit
  EXPECT_EQ(exported.meta->refcount.load(), 1);
  IntelMemoryIPCManager::release_block(whole.ipc_meta);
}
