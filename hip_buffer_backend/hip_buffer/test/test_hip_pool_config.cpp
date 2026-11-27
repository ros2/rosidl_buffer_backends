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

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "hip_buffer/hip_buffer_api.hpp"
#include "hip_buffer/hip_buffer_impl.hpp"
#include "hip_buffer/hip_buffer_ipc_manager.hpp"
#include "hip_buffer/hip_memory_pool.hpp"

namespace hip_pool_config_test
{
using hip_buffer_backend::HipMemoryPool;
using hip_buffer_backend::HipPoolConfig;
using hip_buffer_backend::IPCMetadata;
using hip_buffer_backend::VmmBlock;

constexpr size_t kTestAllocationSize = 64;
constexpr uint64_t kVeryLongGracePeriodUs = std::numeric_limits<uint64_t>::max();

void check_hip(hipError_t result, const char * operation)
{
  if (result != hipSuccess) {
    const char * error_name = hipGetErrorName(result);
    throw std::runtime_error(
            std::string(operation) + " failed: " +
            (error_name ? error_name : "unknown HIP error"));
  }
}

// Do not turn unexpected HIP errors into skips; skip only when this host lacks
// the VMM/FD capability required for these reuse tests.
std::string unavailable_capability()
{
  int count = 0;
  const auto devices = hipGetDeviceCount(&count);
  if (devices == hipErrorNoDevice || devices == hipErrorInsufficientDriver) {
    return hipGetErrorName(devices);
  }
  check_hip(devices, "hipGetDeviceCount");
  if (count == 0) {
    return "no HIP device";
  }

  int device = 0;
  check_hip(hipGetDevice(&device), "hipGetDevice");
  hipMemAllocationProp property{};
  property.type = hipMemAllocationTypePinned;
  property.location.type = hipMemLocationTypeDevice;
  property.location.id = device;
  property.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  size_t granularity = 0;
  const auto granularity_result = hipMemGetAllocationGranularity(
    &granularity, &property, hipMemAllocationGranularityMinimum);
  if (granularity_result == hipErrorNotSupported) {
    return "HIP VMM FD allocation unsupported";
  }
  check_hip(granularity_result, "hipMemGetAllocationGranularity");
  if (granularity == 0) {
    throw std::runtime_error("HIP returned zero VMM granularity");
  }

  hipMemGenericAllocationHandle_t handle{};
  const auto allocation = hipMemCreate(&handle, granularity, &property, 0);
  if (allocation == hipErrorNotSupported) {
    return "HIP VMM allocation unsupported";
  }
  check_hip(allocation, "hipMemCreate(capability probe)");

  int fd = -1;
  const auto exported = hipMemExportToShareableHandle(
    &fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
  if (fd >= 0) {
    close(fd);
  }
  check_hip(hipMemRelease(handle), "hipMemRelease(capability probe)");
  if (exported == hipErrorNotSupported) {
    return "HIP VMM FD export unsupported";
  }
  check_hip(exported, "hipMemExportToShareableHandle");
  return {};
}

// Long grace periods must not make HipMemoryPool's teardown wait for the
// synthetic publication in a test. Declaration order keeps this cleanup ahead
// of pool destruction, including on assertion failures.
class ClearPublicationOnExit
{
public:
  ClearPublicationOnExit() = default;
  explicit ClearPublicationOnExit(VmmBlock * block)
  : block_(block) {}

  ~ClearPublicationOnExit()
  {
    if (block_ && block_->ipc_meta) {
      block_->ipc_meta->uid.store(0, std::memory_order_seq_cst);
      block_->ipc_meta->publish_timestamp_us.store(0, std::memory_order_release);
      block_->current_uid = 0;
    }
  }

  void set_block(VmmBlock * block) {block_ = block;}

  ClearPublicationOnExit(const ClearPublicationOnExit &) = delete;
  ClearPublicationOnExit & operator=(const ClearPublicationOnExit &) = delete;

private:
  VmmBlock * block_{nullptr};
};

// Consumer release matches the production importer: drop the shared IPC lease
// with release ordering after the consumer stops using the imported block.
class ImportedLease
{
public:
  explicit ImportedLease(IPCMetadata * metadata)
  : metadata_(metadata) {}

  ~ImportedLease()
  {
    release();
  }

  ImportedLease(const ImportedLease &) = delete;
  ImportedLease & operator=(const ImportedLease &) = delete;

  int32_t release() noexcept
  {
    if (!metadata_) {
      return 0;
    }
    IPCMetadata * metadata = metadata_;
    metadata_ = nullptr;
    return metadata->refcount.fetch_sub(1, std::memory_order_release);
  }

private:
  IPCMetadata * metadata_;
};

class HipPoolConfigTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    const auto unavailable = unavailable_capability();
    if (!unavailable.empty()) {
      GTEST_SKIP() << unavailable;
    }
  }

  std::shared_ptr<HipMemoryPool> create_pool(const HipPoolConfig & config)
  {
    auto pool = std::make_shared<HipMemoryPool>(config);
    const hipError_t result = pool->create();
    if (result != hipSuccess) {
      ADD_FAILURE() << "HipMemoryPool::create failed: " << hipGetErrorName(result);
      return nullptr;
    }
    if (!pool->is_ipc_capable()) {
      ADD_FAILURE() << "VMM FD capability probe succeeded but the pool disabled IPC";
      return nullptr;
    }
    return pool;
  }
};

TEST_F(HipPoolConfigTest, ZeroGraceAllowsImmediateReuse)
{
  auto pool = create_pool(HipPoolConfig{0});
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->get_config().grace_period_us, uint64_t{0});

  VmmBlock * published = pool->allocate(kTestAllocationSize);
  ClearPublicationOnExit clear_publication(published);
  ASSERT_NE(published->ipc_meta, nullptr);
  ASSERT_NE(pool->assign_uid(published), uint64_t{0});
  ASSERT_NE(published->ipc_meta->publish_timestamp_us.load(std::memory_order_acquire), 0u);

  pool->free(published);
  VmmBlock * reused = pool->allocate(kTestAllocationSize);
  EXPECT_EQ(reused, published);
  pool->free(reused);
}

TEST_F(HipPoolConfigTest, LongGracePreventsReuseWithoutWaiting)
{
  const HipPoolConfig config{kVeryLongGracePeriodUs};
  auto pool = create_pool(config);
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->get_config().grace_period_us, kVeryLongGracePeriodUs);

  VmmBlock * published = pool->allocate(kTestAllocationSize);
  ClearPublicationOnExit clear_publication(published);
  ASSERT_NE(published->ipc_meta, nullptr);
  ASSERT_NE(pool->assign_uid(published), uint64_t{0});

  pool->free(published);
  VmmBlock * replacement = pool->allocate(kTestAllocationSize);
  EXPECT_NE(replacement, published);
  pool->free(replacement);
}

TEST_F(HipPoolConfigTest, ZeroGraceStillWaitsForImportedConsumerLease)
{
  auto pool = create_pool(HipPoolConfig{0});
  ASSERT_NE(pool, nullptr);

  VmmBlock * published = pool->allocate(kTestAllocationSize);
  ClearPublicationOnExit clear_publication(published);
  ASSERT_NE(published->ipc_meta, nullptr);
  const std::string socket_path = pool->register_block_for_ipc(published);
  ASSERT_FALSE(socket_path.empty());
  const uint64_t uid = pool->assign_uid(published);
  ASSERT_NE(uid, uint64_t{0});

  const auto imported = hip_buffer_backend::HipVmmIPCManager::import_block(
    socket_path, static_cast<int32_t>(getpid()), published->block_id, published->size, uid);
  ASSERT_NE(imported.ipc_meta, nullptr);
  ImportedLease lease(imported.ipc_meta);
  EXPECT_EQ(published->ipc_meta->refcount.load(std::memory_order_acquire), 1);

  pool->free(published);
  VmmBlock * while_leased = pool->allocate(kTestAllocationSize);
  EXPECT_NE(while_leased, published);
  EXPECT_EQ(published->ipc_meta->refcount.load(std::memory_order_acquire), 1);

  EXPECT_EQ(lease.release(), 1);
  EXPECT_EQ(published->ipc_meta->refcount.load(std::memory_order_acquire), 0);
  VmmBlock * after_release = pool->allocate(kTestAllocationSize);
  EXPECT_EQ(after_release, published);

  pool->free(after_release);
  pool->free(while_leased);
}

TEST_F(HipPoolConfigTest, GlobalConfigurationAppliesAndRejectsPostInitializationChange)
{
  const HipPoolConfig config{0};
  ASSERT_NO_THROW(hip_buffer_backend::configure_global_pool(config));

  std::shared_ptr<HipMemoryPool> pool;
  VmmBlock * published = nullptr;
  ClearPublicationOnExit clear_publication;
  {
    auto first = hip_buffer_backend::allocate_buffer(kTestAllocationSize);
    auto * implementation = hip_buffer_backend::detail::hip_impl_of(first);
    ASSERT_NE(implementation, nullptr);

    pool = hip_buffer_backend::get_or_create_global_pool();
    ASSERT_NE(pool, nullptr);
    ASSERT_TRUE(pool->is_ipc_capable());
    EXPECT_EQ(pool->get_config().grace_period_us, uint64_t{0});

    published = pool->find_block_for_va(
      implementation->get_hip_buffer().get_device_ptr());
    ASSERT_NE(published, nullptr);
    clear_publication.set_block(published);
    ASSERT_NE(published->ipc_meta, nullptr);
    ASSERT_NE(pool->assign_uid(published), uint64_t{0});

    {
      auto write_handle = implementation->get_hip_buffer().get_write_handle(nullptr);
      (void)write_handle;
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  }

  {
    auto second = hip_buffer_backend::allocate_buffer(kTestAllocationSize);
    auto * second_implementation = hip_buffer_backend::detail::hip_impl_of(second);
    ASSERT_NE(second_implementation, nullptr);
    VmmBlock * reused = pool->find_block_for_va(
      second_implementation->get_hip_buffer().get_device_ptr());
    EXPECT_EQ(reused, published);
  }

  EXPECT_THROW(
    hip_buffer_backend::configure_global_pool(HipPoolConfig{1}), std::logic_error);
  EXPECT_EQ(pool->get_config().grace_period_us, uint64_t{0});
}

}  // namespace hip_pool_config_test

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
