// Copyright 2026 Open Source Robotics Foundation, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <atomic>
#include <utility>
#include <vector>

#include "hip_lifecycle_process.hpp"

namespace hip_lifecycle_test
{
using hip_buffer_backend::HipBuffer;
using hip_buffer_backend::HipBufferBackend;
using hip_buffer_backend::HipMemoryPool;
using hip_buffer_backend::VmmBlock;

// Probe the HIP operations directly: the pool's boolean capability probe hides
// error codes, and must not turn allocation/driver/API failures into test skips.
std::string unavailable_capability()
{
  int count = 0;
  const auto devices = hipGetDeviceCount(&count);
  if (devices == hipErrorNoDevice || devices == hipErrorInsufficientDriver) {
    return hipGetErrorName(devices);
  }
  check_hip(devices, "hipGetDeviceCount");
  if (count == 0) {return "no HIP device";}
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
  if (granularity_result == hipErrorNotSupported) {return "HIP VMM FD allocation unsupported";}
  check_hip(granularity_result, "hipMemGetAllocationGranularity");
  require(granularity != 0, "HIP returned zero VMM granularity");
  hipMemGenericAllocationHandle_t handle{};
  const auto allocation = hipMemCreate(&handle, granularity, &property, 0);
  if (allocation == hipErrorNotSupported) {return "HIP VMM allocation unsupported";}
  check_hip(allocation, "hipMemCreate(capability probe)");
  int fd = -1;
  const auto exported = hipMemExportToShareableHandle(
    &fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
  if (fd >= 0) {close(fd);}
  check_hip(hipMemRelease(handle), "hipMemRelease(capability probe)");
  if (exported == hipErrorNotSupported) {return "HIP VMM FD export unsupported";}
  check_hip(exported, "hipMemExportToShareableHandle");
  return {};
}

class HipBufferLifecycle : public ::testing::Test
{
protected:
  void SetUp() override
  {
    const auto unavailable = unavailable_capability();
    if (!unavailable.empty()) {GTEST_SKIP() << unavailable;}
    pool_ = hip_buffer_backend::get_or_create_global_pool();
    ASSERT_TRUE(pool_->is_ipc_capable()) << "VMM probe succeeded but backend disabled IPC";
    endpoint_.endpoint_type = RMW_ENDPOINT_PUBLISHER;
    endpoint_.topic_type = "hip_buffer_lifecycle_test";
    static uint32_t sequence = 0;
    const uint64_t identity = (static_cast<uint64_t>(getpid()) << 32) | ++sequence;
    static_assert(sizeof(endpoint_.endpoint_gid) >= sizeof(identity), "GID too small");
    std::memcpy(endpoint_.endpoint_gid, &identity, sizeof(identity));
    backend_.on_creating_endpoint(endpoint_);
    ASSERT_TRUE(backend_.on_discovering_endpoint(endpoint_, {}, {{"hip", ""}}).first);
    source_ = allocate();
    original_ = block(*source_);
    ASSERT_NE(original_, nullptr);
    ASSERT_NE(original_->ipc_meta, nullptr);
    write_and_verify(*source_, 0x11);
    old_ = publish(*source_);
    ASSERT_EQ(old_.pid, getpid());
    ASSERT_EQ(old_.block, original_->block_id);
    ASSERT_EQ(old_.block_size, original_->size);
    ASSERT_NE(old_.uid, 0u);
    ASSERT_EQ(original_->ipc_meta->uid.load(), old_.uid);
    ASSERT_EQ(refs(), 0);
    ASSERT_NE(original_->ipc_meta->publish_timestamp_us.load(), 0u);
  }

  std::unique_ptr<Impl> allocate()
  {
    auto * allocated = pool_->allocate(kPayloadSize);
    HipBuffer buffer(allocated->va, kPayloadSize, pool_->get_device_id(), pool_->deleter(allocated));
    return std::make_unique<Impl>(std::move(buffer), kPayloadSize);
  }

  VmmBlock * block(Impl & impl)
  {
    return pool_->find_block_for_va(impl.get_hip_buffer().get_device_ptr());
  }

  WireDescriptor publish(Impl & impl)
  {
    const auto opaque = backend_.create_descriptor_with_endpoint(&impl, endpoint_);
    require(opaque != nullptr, "backend descriptor creation failed");
    return to_wire(*static_cast<Descriptor *>(opaque.get()));
  }

  void write_and_verify(Impl & impl, uint8_t value)
  {
    std::array<uint8_t, kPayloadSize> bytes;
    bytes.fill(value);
    void * pointer = impl.get_hip_buffer().get_device_ptr();
    check_hip(hipMemcpy(pointer, bytes.data(), bytes.size(), hipMemcpyHostToDevice),
      "hipMemcpy(payload write)");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(payload write)");
    bytes.fill(0);
    check_hip(hipMemcpy(bytes.data(), pointer, bytes.size(), hipMemcpyDeviceToHost),
      "hipMemcpy(publisher readback)");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(publisher readback)");
    require(std::all_of(bytes.begin(), bytes.end(),
      [value](uint8_t byte) {return byte == value;}), "publisher payload readback mismatch");
  }

  int32_t refs() const {return original_->ipc_meta->refcount.load(std::memory_order_acquire);}

  void baseline(Importer & importer)
  {
    const auto reply = importer.request(Command::Import, old_, 0x11);
    require(reply.size == kPayloadSize && reply.payload_matches,
      "normal baseline import did not read all 4096 bytes of 0x11");
    require(refs() == 1, "normal import did not acquire exactly one shared reference");
  }

  void release(Importer & importer)
  {
    importer.request(Command::Release);
    require(refs() == 0, "import release did not balance the shared reference");
  }

  void return_source_after_grace()
  {
    const auto published = original_->ipc_meta->publish_timestamp_us.load(std::memory_order_acquire);
    require(published != 0, "missing publication timestamp before grace wait");
    source_.reset();
    const auto reusable_at = std::chrono::steady_clock::time_point(
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::microseconds(published) +
        std::chrono::microseconds(pool_->get_config().grace_period_us)));
    require(reusable_at <= std::chrono::steady_clock::now() + kPhaseTimeout,
      "publication timestamp is outside the phase deadline");
    std::this_thread::sleep_until(reusable_at);
  }

  std::unique_ptr<Impl> reuse_original()
  {
    // Earlier tests can leave other free blocks in the process-global pool.
    // Hold competing allocations so reuse is observed, never assumed or skipped.
    for (size_t attempt = 0; attempt < 32; ++attempt) {
      auto candidate = allocate();
      if (block(*candidate) == original_) {
        require(original_->block_id == old_.block, "reused block identity changed");
        require(original_->current_uid == 0, "allocation already received a publication UID");
        return candidate;
      }
      competitors_.push_back(std::move(candidate));
    }
    throw std::runtime_error("original VMM block was not reused within 32 allocations");
  }

  void expect_rejected(Importer & importer)
  {
    const auto reply = importer.request(Command::Import, old_, 0xa5);
    EXPECT_EQ(reply.size, 0u)
      << "stale descriptor accepted; first byte=" << static_cast<unsigned>(reply.first_byte)
      << ", entire replacement payload=" << reply.payload_matches;
    EXPECT_EQ(refs(), 0) << "rejected import must roll back its reference increment";
    release(importer);
  }

  std::shared_ptr<HipMemoryPool> pool_;
  HipBufferBackend backend_;
  rmw_topic_endpoint_info_t endpoint_{};
  std::unique_ptr<Impl> source_;
  std::vector<std::unique_ptr<Impl>> competitors_;
  VmmBlock * original_{};
  WireDescriptor old_{};
};

TEST_F(HipBufferLifecycle, NormalImportReadsOriginalPayload)
{
  Importer importer;
  baseline(importer);
  release(importer);
  importer.finish();
}

TEST_F(HipBufferLifecycle, ImportedReferenceBlocksReuseUntilReleased)
{
  Importer importer;
  baseline(importer);
  return_source_after_grace();
  auto while_held = allocate();
  ASSERT_NE(block(*while_held), original_);
  ASSERT_NE(block(*while_held)->block_id, old_.block);
  ASSERT_EQ(refs(), 1);
  const auto held = importer.request(Command::ReadHeld, {}, 0x11);
  ASSERT_EQ(held.size, kPayloadSize);
  ASSERT_TRUE(held.payload_matches) << "held import changed while a different block was allocated";
  ASSERT_EQ(refs(), 1);
  release(importer);
  auto reused = reuse_original();
  ASSERT_EQ(block(*reused), original_);
  ASSERT_EQ(refs(), 0);
  importer.finish();
}

TEST_F(HipBufferLifecycle, OldDescriptorRejectedAfterReuseAndNewUid)
{
  Importer cached;
  baseline(cached);
  release(cached);
  return_source_after_grace();
  auto reused = reuse_original();
  write_and_verify(*reused, 0xa5);
  const auto replacement = publish(*reused);
  ASSERT_EQ(replacement.block, old_.block);
  ASSERT_NE(replacement.uid, old_.uid);
  ASSERT_EQ(original_->ipc_meta->uid.load(), replacement.uid);
  expect_rejected(cached);
  Importer uncached;
  expect_rejected(uncached);
  const auto valid = uncached.request(Command::Import, replacement, 0xa5);
  EXPECT_EQ(valid.size, kPayloadSize);
  EXPECT_TRUE(valid.payload_matches);
  EXPECT_EQ(refs(), 1);
  release(uncached);
  uncached.finish();
  cached.finish();
}

TEST_F(HipBufferLifecycle, OldDescriptorRejectedAfterReuseBeforeNewUid)
{
  Importer cached;
  baseline(cached);
  release(cached);
  return_source_after_grace();
  auto reused = reuse_original();
  write_and_verify(*reused, 0xa5);
  ASSERT_EQ(original_->current_uid, 0u);
  ASSERT_EQ(original_->ipc_meta->publish_timestamp_us.load(), 0u);
  // Do not publish or assign a new UID here. Correct behavior must reject the
  // previous incarnation as soon as its block is reused, even before republish.
  expect_rejected(cached);
  Importer uncached;
  expect_rejected(uncached);
  uncached.finish();
  cached.finish();
}
}  // namespace hip_lifecycle_test

int main(int argc, char ** argv)
{
  if (argc == 2 && std::strcmp(argv[1], "--lifecycle-importer") == 0) {
    return hip_lifecycle_test::importer_main(3);
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
