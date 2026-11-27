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
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "intel_buffer/intel_buffer_api.hpp"
#include "intel_buffer_backend/intel_buffer_backend.hpp"
#include "intel_buffer_backend_msgs/msg/intel_buffer_descriptor.hpp"
#include "rmw/topic_endpoint_info.h"

using intel_buffer_backend::IntelBufferBackend;
using intel_buffer_backend::IntelBufferImpl;
using IntelBufferDescriptor = intel_buffer_backend_msgs::msg::IntelBufferDescriptor;

namespace
{
rmw_topic_endpoint_info_t make_endpoint(uint32_t gid_seed)
{
  rmw_topic_endpoint_info_t info = rmw_get_zero_initialized_topic_endpoint_info();
  info.endpoint_gid[0] = static_cast<uint8_t>((gid_seed >> 24) & 0xFF);
  info.endpoint_gid[1] = static_cast<uint8_t>((gid_seed >> 16) & 0xFF);
  info.endpoint_gid[2] = static_cast<uint8_t>((gid_seed >> 8) & 0xFF);
  info.endpoint_gid[3] = static_cast<uint8_t>(gid_seed & 0xFF);
  return info;
}

// The trailing "|host=...|pidns=..." entries this backend advertises.
std::string local_host_entry(const IntelBufferBackend & backend)
{
  std::string meta = backend.get_backend_metadata();
  return meta.substr(meta.find("|host="));
}
}  // namespace

TEST(IntelBufferBackend, BasicIdentityAccessors)
{
  IntelBufferBackend backend;
  EXPECT_EQ(backend.get_backend_type(), "intel_buffer");
  EXPECT_NE(backend.get_descriptor_type_support(), nullptr);

  auto descriptor = backend.create_empty_descriptor();
  ASSERT_NE(descriptor, nullptr);
  EXPECT_NE(std::static_pointer_cast<IntelBufferDescriptor>(descriptor), nullptr);
}

TEST(IntelBufferBackend, BackendMetadataAlwaysAdvertisesExtmemmap)
{
  IntelBufferBackend backend;
  std::string meta = backend.get_backend_metadata();
  auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();

  if (!pool) {
    EXPECT_EQ(meta.rfind("transport=extmemmap|extmemmap=1|host=", 0), 0u);
  } else {
    EXPECT_NE(meta.find("transport="), std::string::npos);
    EXPECT_NE(meta.find("device="), std::string::npos);
    EXPECT_NE(meta.find("extmemmap=1"), std::string::npos);
  }
  // Non-empty host identity
  EXPECT_GT(local_host_entry(backend).size(), std::string("|host=").size());
  // ',' and ';' are RMW user_data delimiters; the value is truncated at either
  EXPECT_EQ(meta.find_first_of(",;"), std::string::npos);
}

TEST(IntelBufferBackend, BackendMetadataAdvertisesActivePoolTransport)
{
  IntelBufferBackend backend;
  auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
  if (!pool || !IntelBufferImpl<uint8_t>::is_ipc_capable()) {
    GTEST_SKIP() << "No IPC-capable pool transport on this machine";
  }
  const std::string expected =
    pool->transport_mode() == intel_buffer_backend::TransportMode::LEVEL_ZERO_USM ?
    "transport=l0usm|" : "transport=dmabuf|";
  EXPECT_EQ(backend.get_backend_metadata().rfind(expected, 0), 0u);
}

TEST(IntelBufferBackendDiscovery, MissingBackendKeyIsIncompatible)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(1);
  std::unordered_map<std::string, std::string> supported;  // no "intel_buffer" key

  auto result = backend.on_discovering_endpoint(endpoint, {}, supported);
  EXPECT_FALSE(result.first);
}

TEST(IntelBufferBackendDiscovery, ExtmemmapAdvertisedRemoteIsCompatible)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(2);
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] = "transport=extmemmap|extmemmap=1" + local_host_entry(backend);

  auto result = backend.on_discovering_endpoint(endpoint, {}, supported);
  EXPECT_TRUE(result.first);
}

TEST(IntelBufferBackendDiscovery, PeerOnAnotherHostIsIncompatible)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(13);
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] =
    "transport=extmemmap|extmemmap=1|host=00000000-0000-0000-0000-000000000000";

  auto result = backend.on_discovering_endpoint(endpoint, {}, supported);
  EXPECT_FALSE(result.first);
}

// The "|host=..." entry alone, without the entries that follow it.
std::string local_host_only(const IntelBufferBackend & backend)
{
  const std::string entry = local_host_entry(backend);
  return entry.substr(0, entry.find('|', 1));
}

// A same-host peer in another PID namespace (another container) can't share
// pid-tracked references with this process: it gets the CPU-based message.
TEST(IntelBufferBackendDiscovery, PeerInAnotherPidNamespaceIsIncompatible)
{
  IntelBufferBackend backend;
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] =
    "transport=dmabuf|device=0|extmemmap=1" + local_host_only(backend) + "|pidns=1";
  EXPECT_FALSE(backend.on_discovering_endpoint(make_endpoint(16), {}, supported).first);

  supported["intel_buffer"] = "transport=dmabuf|device=0|extmemmap=1" + local_host_only(backend);
  EXPECT_FALSE(backend.on_discovering_endpoint(make_endpoint(17), {}, supported).first) <<
    "a peer that doesn't advertise its PID namespace";
}

TEST(IntelBufferBackendDiscovery, PeerWithoutHostIsIncompatible)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(14);
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] = "transport=extmemmap|extmemmap=1";

  auto result = backend.on_discovering_endpoint(endpoint, {}, supported);
  EXPECT_FALSE(result.first);
}

TEST(IntelBufferBackendDescriptor, PeerOnAnotherHostGetsNoDescriptor)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(15);
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] =
    "transport=extmemmap|extmemmap=1|host=00000000-0000-0000-0000-000000000000";
  ASSERT_FALSE(backend.on_discovering_endpoint(endpoint, {}, supported).first);

  auto block = std::make_unique<intel_buffer_backend::PoolBlock>();
  uint8_t byte = 0;
  block->ptr = &byte;
  block->size = 1;
  block->transport = intel_buffer_backend::TransportMode::EXTERNAL_MEMMAP;
  block->extmap_shm_name = "/not_used";

  IntelBufferImpl<uint8_t> impl(block.release(), 1, false);
  EXPECT_EQ(backend.create_descriptor_with_endpoint(&impl, endpoint), nullptr);
}

TEST(IntelBufferBackendDiscovery, MatchingPoolTransportIsCompatible)
{
  IntelBufferBackend backend;
  auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
  if (!pool || !IntelBufferImpl<uint8_t>::is_ipc_capable()) {
    GTEST_SKIP() << "No IPC-capable pool transport on this machine";
  }

  auto endpoint = make_endpoint(3);
  std::unordered_map<std::string, std::string> supported;
  std::string remote_transport =
    pool->is_level_zero_capable() ? "transport=l0usm" : "transport=dmabuf";
  supported["intel_buffer"] = remote_transport + local_host_entry(backend);

  auto result = backend.on_discovering_endpoint(endpoint, {}, supported);
  EXPECT_TRUE(result.first);
}

TEST(IntelBufferBackendDiscovery, MismatchedTransportIsIncompatible)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(4);
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] = "transport=none";

  auto result = backend.on_discovering_endpoint(endpoint, {}, supported);
  EXPECT_FALSE(result.first);
}

TEST(IntelBufferBackendDiscovery, ExtmemmapCapabilityDoesNotOverridePoolTransportMismatch)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(12);
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] = "transport=bogus|extmemmap=1";

  auto result = backend.on_discovering_endpoint(endpoint, {}, supported);
  EXPECT_FALSE(result.first);
}

TEST(IntelBufferBackendDescriptor, NullImplPointerReturnsNullptr)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(5);
  IntelBufferImpl<uint8_t> empty_impl;
  auto descriptor = backend.create_descriptor_with_endpoint(&empty_impl, endpoint);
  EXPECT_EQ(descriptor, nullptr);
}

TEST(IntelBufferBackendDescriptor, ExternalMemmapBlockWithoutShmNameReturnsNullptr)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(6);

  auto block = std::make_unique<intel_buffer_backend::PoolBlock>();
  uint8_t byte = 0;
  block->ptr = &byte;
  block->size = 1;
  block->transport = intel_buffer_backend::TransportMode::EXTERNAL_MEMMAP;
  block->extmap_shm_name = "";

  IntelBufferImpl<uint8_t> impl(block.release(), 1, false);
  auto descriptor = backend.create_descriptor_with_endpoint(&impl, endpoint);
  EXPECT_EQ(descriptor, nullptr);
}

// GIDs of different endpoints can share bytes (a DDS GUID prefix holds the
// vendor and host id); each endpoint keeps its own zero-copy decision regardless.
TEST(IntelBufferBackendDescriptor, IpcDecisionIsPerEndpointNotPerGidPrefix)
{
  IntelBufferBackend backend;
  auto compatible = make_endpoint(0x010f1234);
  auto incompatible = make_endpoint(0x010f1234);
  compatible.endpoint_gid[15] = 1;
  incompatible.endpoint_gid[15] = 2;

  std::unordered_map<std::string, std::string> same_host;
  same_host["intel_buffer"] = "transport=extmemmap|extmemmap=1" + local_host_entry(backend);
  std::unordered_map<std::string, std::string> other_host;
  other_host["intel_buffer"] =
    "transport=extmemmap|extmemmap=1|host=00000000-0000-0000-0000-000000000000";
  ASSERT_TRUE(backend.on_discovering_endpoint(compatible, {}, same_host).first);
  ASSERT_FALSE(backend.on_discovering_endpoint(incompatible, {}, other_host).first);

  auto block = std::make_unique<intel_buffer_backend::PoolBlock>();
  uint8_t byte = 0;
  block->ptr = &byte;
  block->size = 1;
  block->transport = intel_buffer_backend::TransportMode::EXTERNAL_MEMMAP;
  block->extmap_shm_name = "/not_used";
  IntelBufferImpl<uint8_t> impl(block.release(), 1, false);

  EXPECT_NE(backend.create_descriptor_with_endpoint(&impl, compatible), nullptr);
  EXPECT_EQ(backend.create_descriptor_with_endpoint(&impl, incompatible), nullptr);
}

// A peer without Level Zero (it advertises dmabuf) cannot import Level Zero USM
// handles, but can still take transports that need no Level Zero.
TEST(IntelBufferBackendDescriptor, LevelZeroBlockNotSentToPeerWithoutLevelZero)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(16);
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] = "transport=dmabuf|device=0|extmemmap=1" + local_host_entry(backend);
  ASSERT_TRUE(backend.on_discovering_endpoint(endpoint, {}, supported).first);

  uint8_t byte = 0;
  auto extmap_block = std::make_unique<intel_buffer_backend::PoolBlock>();
  extmap_block->ptr = &byte;
  extmap_block->size = 1;
  extmap_block->transport = intel_buffer_backend::TransportMode::EXTERNAL_MEMMAP;
  extmap_block->extmap_shm_name = "/not_used";
  IntelBufferImpl<uint8_t> extmap_impl(extmap_block.release(), 1, false);
  EXPECT_NE(backend.create_descriptor_with_endpoint(&extmap_impl, endpoint), nullptr);

  // A real, exportable Level Zero USM block: an "l0usm" peer gets its descriptor.
  auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
  if (!pool || pool->transport_mode() != intel_buffer_backend::TransportMode::LEVEL_ZERO_USM) {
    GTEST_SKIP() << "pool does not allocate Level Zero USM on this machine";
  }
  auto buffer = intel_buffer_backend::allocate_buffer(4096);
  auto l0_peer = make_endpoint(17);
  std::unordered_map<std::string, std::string> l0_supported;
  l0_supported["intel_buffer"] = "transport=l0usm|device=0|extmemmap=1" + local_host_entry(backend);
  ASSERT_TRUE(backend.on_discovering_endpoint(l0_peer, {}, l0_supported).first);
  if (!backend.create_descriptor_with_endpoint(buffer.get_impl(), l0_peer)) {
    GTEST_SKIP() << "pool block not exportable on this machine";
  }
  EXPECT_EQ(backend.create_descriptor_with_endpoint(buffer.get_impl(), endpoint), nullptr);
}

TEST(IntelBufferBackendDescriptor, IpcDecisionCacheShortCircuitsToNullptr)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(7);
  std::unordered_map<std::string, std::string> supported;  // missing key -> incompatible
  auto discover_result = backend.on_discovering_endpoint(endpoint, {}, supported);
  ASSERT_FALSE(discover_result.first);

  IntelBufferImpl<uint8_t> empty_impl;
  auto descriptor = backend.create_descriptor_with_endpoint(&empty_impl, endpoint);
  EXPECT_EQ(descriptor, nullptr);
}

TEST(IntelBufferBackendFromDescriptor, GarbageDmabufDescriptorReturnsEmptyBufferNotThrow)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(8);

  auto frame = std::make_shared<IntelBufferDescriptor>();
  frame->transport_mode = IntelBufferDescriptor::TRANSPORT_DMA_BUF;
  frame->dmabuf_desc.size = 4096;
  frame->dmabuf_desc.local_ptr = 0;  // force the IPC-import path, not the shortcut
  frame->dmabuf_desc.publisher_pid = 999999;
  frame->dmabuf_desc.dmabuf_pid = 999999;
  frame->dmabuf_desc.dmabuf_pool_block_id = 42;
  frame->dmabuf_desc.dmabuf_socket_path = "/tmp/does_not_exist_backend_test.sock";
  frame->dmabuf_desc.ipc_uid = 1;
  frame->dmabuf_desc.pool_id = 0;

  std::unique_ptr<void, void (*)(void *)> result{nullptr, [](void *) {}};
  ASSERT_NO_THROW(
    result = backend.from_descriptor_with_endpoint(frame.get(), endpoint));
  ASSERT_NE(result.get(), nullptr);
  auto * impl = static_cast<rosidl::BufferImplBase<uint8_t> *>(result.get());
  EXPECT_EQ(impl->size(), 0u);
}

// local_ptr and publisher_pid are untrusted wire data: any publisher on the
// topic can name this process's pid and an arbitrary address. The intra-process
// shortcut must only wrap memory this process verifiably published.
TEST(IntelBufferBackendFromDescriptor, ForgedIntraProcessLocalPtrIsNotTrusted)
{
  uint8_t secret[64];
  std::memset(secret, 0x9, sizeof(secret));
  IntelBufferBackend backend;

  for (uint8_t transport : {IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM,
      IntelBufferDescriptor::TRANSPORT_DMA_BUF})
  {
    IntelBufferDescriptor frame;
    frame.transport_mode = transport;
    if (transport == IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM) {
      frame.l0_desc.size = sizeof(secret);
      frame.l0_desc.local_ptr = reinterpret_cast<uint64_t>(secret);
      frame.l0_desc.publisher_pid = static_cast<int32_t>(getpid());
    } else {
      frame.dmabuf_desc.size = sizeof(secret);
      frame.dmabuf_desc.dmabuf_block_size = sizeof(secret);
      frame.dmabuf_desc.local_ptr = reinterpret_cast<uint64_t>(secret);
      frame.dmabuf_desc.publisher_pid = static_cast<int32_t>(getpid());
    }
    auto result = backend.from_descriptor_with_endpoint(&frame, make_endpoint(9));
    auto * impl = static_cast<IntelBufferImpl<uint8_t> *>(result.get());
    EXPECT_NE(impl->data(), secret) << "transport " << int(transport)
                                    << ": wrapped a forged local_ptr";
  }
}

TEST(IntelBufferBackendFromDescriptor, OwnPoolBlockTakesIntraProcessShortcut)
{
  auto buffer = intel_buffer_backend::allocate_buffer(4096);
  uint8_t * ptr = intel_buffer_backend::get_usm_ptr(buffer);
  if (ptr == nullptr) {
    GTEST_SKIP() << "no buffer pool on this machine";
  }
  ptr[0] = 0x6b;
  IntelBufferBackend backend;
  auto desc = std::static_pointer_cast<IntelBufferDescriptor>(
    backend.create_descriptor_with_endpoint(buffer.get_impl(), make_endpoint(30)));
  if (!desc) {
    GTEST_SKIP() << "pool block not exportable on this machine";
  }

  auto result = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(31));
  auto * impl = static_cast<IntelBufferImpl<uint8_t> *>(result.get());
  EXPECT_EQ(impl->data(), ptr);
  EXPECT_EQ(impl->size(), 4096u);

  // Same block, but not the publication it was handed out for (stale or
  // guessed uid): no shortcut.
  auto stale = *desc;
  if (stale.transport_mode == IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM) {
    stale.l0_desc.ipc_uid ^= 1;
  } else {
    stale.dmabuf_desc.ipc_uid ^= 1;
  }
  auto stale_result = backend.from_descriptor_with_endpoint(&stale, make_endpoint(32));
  EXPECT_NE(static_cast<IntelBufferImpl<uint8_t> *>(stale_result.get())->data(), ptr);
}

// --- External-memmap ring slots imported through the backend ---------------

namespace
{
using intel_buffer_backend::ExternalMemMapRing;
constexpr size_t kRingBytes = 4096;

// Descriptor the backend would publish for `slot` (producer side).
std::shared_ptr<IntelBufferDescriptor> describe_ring_slot(
  IntelBufferBackend & backend, ExternalMemMapRing & ring, size_t slot)
{
  auto buffer = ring.wrap(slot, kRingBytes);
  return std::static_pointer_cast<IntelBufferDescriptor>(
    backend.create_descriptor_with_endpoint(buffer.get_impl(), make_endpoint(20)));
}

IntelBufferImpl<uint8_t> * as_impl(const std::unique_ptr<void, void (*)(void *)> & p)
{
  return static_cast<IntelBufferImpl<uint8_t> *>(p.get());
}
}  // namespace

TEST(IntelBufferBackendExternalMemmap, IntraProcessImportReadsProducerSlotAndHoldsIt)
{
  IntelBufferBackend backend;
  ExternalMemMapRing ring;
  ASSERT_TRUE(ring.create("/ibb_plugin_extmap_intra", kRingBytes, 2));
  const size_t slot = ring.advance();
  ring.data(slot)[0] = 0x42;
  auto desc = describe_ring_slot(backend, ring, slot);
  ASSERT_NE(desc, nullptr);

  auto result = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(21));
  auto * impl = as_impl(result);
  ASSERT_NE(impl->get_block(), nullptr);
  EXPECT_EQ(impl->size(), kRingBytes);
  EXPECT_EQ(impl->data(), ring.data(slot));
  EXPECT_EQ(impl->data()[0], 0x42);
  EXPECT_EQ(impl->get_block()->size, ring.slot_size());
  EXPECT_EQ(impl->get_block()->extmap_generation, desc->extmap_desc.slot_generation);
  EXPECT_NE(impl->get_block()->ipc_meta, nullptr);
}

TEST(IntelBufferBackendExternalMemmap, ForgedLocalPtrForGenuineSlotIsNotTrusted)
{
  IntelBufferBackend backend;
  ExternalMemMapRing ring;
  ASSERT_TRUE(ring.create("/ibb_plugin_extmap_forged", kRingBytes, 1));
  const size_t slot = ring.advance();
  ring.data(slot)[0] = 0x42;
  auto desc = describe_ring_slot(backend, ring, slot);
  uint8_t secret[kRingBytes] = {};
  desc->extmap_desc.local_ptr = reinterpret_cast<uint64_t>(secret);

  auto result = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(29));
  auto * impl = as_impl(result);
  ASSERT_NE(impl->data(), nullptr);
  EXPECT_NE(impl->data(), secret);
  EXPECT_EQ(impl->data()[0], 0x42);  // still reads the genuine slot
}

TEST(IntelBufferBackendExternalMemmap, HeldSlotIsNotReusedUntilReleased)
{
  IntelBufferBackend backend;
  ExternalMemMapRing ring;
  ASSERT_TRUE(ring.create("/ibb_plugin_extmap_held", kRingBytes, 2));
  const size_t held = ring.advance();
  auto desc = describe_ring_slot(backend, ring, held);
  auto result = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(22));
  ASSERT_NE(as_impl(result)->data(), nullptr);

  const size_t other = ring.advance();
  EXPECT_NE(other, held);
  EXPECT_EQ(ring.advance(), other) << "advance() handed out a slot a consumer holds";

  result.reset();  // consumer releases the message
  EXPECT_EQ(ring.advance(), held);
}

TEST(IntelBufferBackendExternalMemmap, AllSlotsHeldReturnsNoFreeSlot)
{
  IntelBufferBackend backend;
  ExternalMemMapRing ring;
  ASSERT_TRUE(ring.create("/ibb_plugin_extmap_full", kRingBytes, 1));
  const size_t slot = ring.advance();
  auto desc = describe_ring_slot(backend, ring, slot);
  auto result = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(23));
  ASSERT_NE(as_impl(result)->data(), nullptr);

  EXPECT_EQ(ring.advance(), ExternalMemMapRing::kNoFreeSlot);
  result.reset();
  EXPECT_EQ(ring.advance(), slot);
}

TEST(IntelBufferBackendExternalMemmap, StaleDescriptorIsRejectedAndHoldsNothing)
{
  IntelBufferBackend backend;
  ExternalMemMapRing ring;
  ASSERT_TRUE(ring.create("/ibb_plugin_extmap_stale", kRingBytes, 1));
  const size_t slot = ring.advance();
  auto desc = describe_ring_slot(backend, ring, slot);
  ASSERT_EQ(ring.advance(), slot);  // producer reuses the slot before import

  auto result = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(24));
  EXPECT_EQ(as_impl(result)->data(), nullptr);
  EXPECT_EQ(as_impl(result)->size(), 0u);
  EXPECT_EQ(ring.advance(), slot) << "rejected import left a reference on the slot";
}

TEST(IntelBufferBackendExternalMemmap, DescriptorWithoutRingSlotIsRejected)
{
  // A local pointer with no shared slot behind it can't be protected from
  // reuse, so the import fails closed instead of wrapping it unprotected.
  uint8_t heap[16] = {};
  IntelBufferBackend backend;
  auto frame = std::make_shared<IntelBufferDescriptor>();
  frame->transport_mode = IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP;
  frame->extmap_desc.size = sizeof(heap);
  frame->extmap_desc.local_ptr = reinterpret_cast<uint64_t>(heap);
  frame->extmap_desc.publisher_pid = static_cast<int32_t>(getpid());

  auto result = backend.from_descriptor_with_endpoint(frame.get(), make_endpoint(25));
  EXPECT_EQ(as_impl(result)->data(), nullptr);
}

TEST(IntelBufferBackendExternalMemmap, SlotHeldByDeadConsumerIsReclaimed)
{
  IntelBufferBackend backend;
  ExternalMemMapRing ring;
  ASSERT_TRUE(ring.create("/ibb_plugin_extmap_dead", kRingBytes, 1));
  const size_t slot = ring.advance();
  auto desc = describe_ring_slot(backend, ring, slot);

  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    // Another process imports the slot, then dies without releasing it.
    auto result = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(26));
    _exit(as_impl(result)->data() != nullptr ? 0 : 1);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "child failed to import";

  EXPECT_EQ(ring.advance(), slot);
}

TEST(IntelBufferBackendExternalMemmap, HeldFrameSurvivesItsProducerDying)
{
  // A subscriber holds a frame, its producer process dies, then the subscriber
  // imports from another producer (which evicts dead producers' regions from
  // the import cache). The held frame must stay mapped: reading it and
  // releasing its slot reference must not touch unmapped memory. Runs as a
  // death test so a fault fails this test instead of the whole binary.
  EXPECT_EXIT(
  {
    int to_parent[2];
    int to_child[2];
    if (pipe(to_parent) != 0 || pipe(to_child) != 0) {
      _exit(10);
    }
    const pid_t producer = fork();
    if (producer == 0) {
      IntelBufferBackend producer_backend;
      ExternalMemMapRing ring;
      if (!ring.create("/ibb_plugin_extmap_dying", kRingBytes, 1)) {
        _exit(1);
      }
      const size_t slot = ring.advance();
      ring.data(slot)[0] = 0x5a;
      auto desc = describe_ring_slot(producer_backend, ring, slot);
      const std::string msg = desc->extmap_desc.shm_name + "\n" +
        std::to_string(desc->extmap_desc.slot_generation) + "\n";
      char go;
      if (write(to_parent[1], msg.data(), msg.size()) != static_cast<ssize_t>(msg.size()) ||
        read(to_child[0], &go, 1) != 1)
      {
        _exit(1);
      }
      _exit(0);  // die without unmapping or unlinking anything
    }
    char buf[256] = {};
    if (producer < 0 || read(to_parent[0], buf, sizeof(buf) - 1) <= 0) {
      _exit(11);
    }
    const std::string reply(buf);
    const std::string shm_name = reply.substr(0, reply.find('\n'));

    IntelBufferBackend backend;
    IntelBufferDescriptor frame;
    frame.transport_mode = IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP;
    frame.extmap_desc.shm_name = shm_name;
    frame.extmap_desc.size = kRingBytes;
    frame.extmap_desc.slot_generation = static_cast<uint32_t>(
      std::stoul(reply.substr(reply.find('\n') + 1)));
    frame.extmap_desc.publisher_pid = producer;
    auto held = backend.from_descriptor_with_endpoint(&frame, make_endpoint(27));
    if (as_impl(held)->data() == nullptr) {
      _exit(12);
    }

    // The producer dies (and is reaped) while the frame is held.
    if (write(to_child[1], "x", 1) != 1 || waitpid(producer, nullptr, 0) != producer) {
      _exit(13);
    }

    // Importing from another producer sweeps dead producers' regions.
    ExternalMemMapRing other;
    if (!other.create("/ibb_plugin_extmap_other", kRingBytes, 1)) {
      _exit(14);
    }
    auto other_desc = describe_ring_slot(backend, other, other.advance());
    other_desc->extmap_desc.publisher_pid = static_cast<int32_t>(getppid());
    auto fresh = backend.from_descriptor_with_endpoint(other_desc.get(), make_endpoint(28));

    const int first_byte = as_impl(held)->data()[0];
    held.reset();  // releases the slot reference in the dead producer's region
    fresh.reset();
    other.close();
    shm_unlink(shm_name.c_str());  // the producer died without cleaning up
    _exit(first_byte == 0x5a ? 0 : 15);
  }, ::testing::ExitedWithCode(0), "");
}

TEST(IntelBufferBackendFromDescriptor, ZeroSizeExternalMemmapDescriptorReturnsEmptyBuffer)
{
  IntelBufferBackend backend;
  auto endpoint = make_endpoint(11);

  auto frame = std::make_shared<IntelBufferDescriptor>();
  frame->transport_mode = IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP;
  frame->extmap_desc.size = 0;

  auto result = backend.from_descriptor_with_endpoint(frame.get(), endpoint);
  ASSERT_NE(result.get(), nullptr);
  auto * impl = static_cast<rosidl::BufferImplBase<uint8_t> *>(result.get());
  EXPECT_EQ(impl->size(), 0u);
}

// --- DMA-BUF imports: sizes in the descriptor are untrusted ------------------

namespace
{
constexpr size_t kDmaBufBytes = 4096;

// An exported buffer of kDmaBufBytes, shared over the real DmaBufIpc socket.
// A memfd stands in for the dma-buf: unlike a dma-buf, mapping it past its end
// succeeds and only faults (SIGBUS) on access, which is what the size checks
// must prevent.
struct ExportedBuffer
{
  // The uid the "producer" published the frame under.
  static constexpr uint64_t kUid = 0x5eed;

  explicit ExportedBuffer(uint32_t block_id)
  : block_id(block_id), fd(memfd_create("ibb_plugin_dmabuf", 0)),
    // Its descriptors use the block id as the pool id too.
    meta_name(intel_memory_core::ipc_meta_shm_name(getpid(), block_id, block_id))
  {
    if (fd >= 0 && ftruncate(fd, kDmaBufBytes) == 0) {
      const uint8_t marker = 0x3c;
      if (pwrite(fd, &marker, 1, 0) == 1) {
        socket_path = intel_buffer_backend::DmaBufIpc::register_fd(block_id, fd);
      }
    }
    // The block's refcount metadata, as the producer's pool creates it.
    const int meta_fd = shm_open(meta_name.c_str(), O_CREAT | O_RDWR, 0600);
    if (meta_fd >= 0 && ftruncate(meta_fd, sizeof(intel_memory_core::IPCMetadata)) == 0) {
      void * p = mmap(nullptr, sizeof(intel_memory_core::IPCMetadata),
          PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd, 0);
      if (p != MAP_FAILED) {
        meta = new (p) intel_memory_core::IPCMetadata();
        meta->uid.store(kUid);
      }
    }
    if (meta_fd >= 0) {
      close(meta_fd);
    }
  }
  ~ExportedBuffer()
  {
    intel_buffer_backend::DmaBufIpc::unregister_block(block_id);
    if (fd >= 0) {
      close(fd);
    }
    if (meta) {
      munmap(meta, sizeof(intel_memory_core::IPCMetadata));
    }
    shm_unlink(meta_name.c_str());
  }

  // Descriptor as a (possibly buggy or hostile) peer could send it.
  IntelBufferDescriptor descriptor(uint32_t pool_id, uint64_t size, uint64_t block_size) const
  {
    IntelBufferDescriptor frame;
    frame.transport_mode = IntelBufferDescriptor::TRANSPORT_DMA_BUF;
    auto & d = frame.dmabuf_desc;
    d.size = size;
    d.dmabuf_block_size = block_size;
    d.dmabuf_socket_path = socket_path;
    d.dmabuf_pid = static_cast<int32_t>(getpid());
    d.dmabuf_pool_block_id = block_id;
    d.pool_id = pool_id;
    d.ipc_uid = kUid;
    d.publisher_pid = static_cast<int32_t>(getpid());  // local_ptr == 0: not intra-process
    return frame;
  }

  uint32_t block_id;
  int fd;
  std::string meta_name;
  intel_memory_core::IPCMetadata * meta{nullptr};
  std::string socket_path;
};
}  // namespace

TEST(IntelBufferBackendDmaBuf, ImportWithConsistentSizesSucceeds)
{
  ExportedBuffer exported(0xd0b0001u);
  ASSERT_FALSE(exported.socket_path.empty());
  IntelBufferBackend backend;
  auto frame = exported.descriptor(0xd0b0001u, kDmaBufBytes, kDmaBufBytes);

  auto result = backend.from_descriptor_with_endpoint(&frame, make_endpoint(40));
  auto * impl = as_impl(result);
  ASSERT_NE(impl->data(), nullptr);
  EXPECT_EQ(impl->size(), kDmaBufBytes);
  EXPECT_EQ(impl->data()[0], 0x3c);
  EXPECT_EQ(impl->data()[kDmaBufBytes - 1], 0);
}

TEST(IntelBufferBackendDmaBuf, PayloadLargerThanMappingIsRejected)
{
  // The subscriber would be handed size bytes over a block_size mapping.
  ExportedBuffer exported(0xd0b0002u);
  ASSERT_FALSE(exported.socket_path.empty());
  IntelBufferBackend backend;
  auto frame = exported.descriptor(0xd0b0002u, 4 * kDmaBufBytes, kDmaBufBytes);

  auto result = backend.from_descriptor_with_endpoint(&frame, make_endpoint(41));
  EXPECT_EQ(as_impl(result)->data(), nullptr);
  EXPECT_EQ(as_impl(result)->size(), 0u);
}

TEST(IntelBufferBackendDmaBuf, MappingLargerThanExportedBufferIsRejected)
{
  // block_size exceeds what the received fd actually holds.
  ExportedBuffer exported(0xd0b0003u);
  ASSERT_FALSE(exported.socket_path.empty());
  IntelBufferBackend backend;
  auto frame = exported.descriptor(0xd0b0003u, 4 * kDmaBufBytes, 4 * kDmaBufBytes);

  auto result = backend.from_descriptor_with_endpoint(&frame, make_endpoint(42));
  EXPECT_EQ(as_impl(result)->data(), nullptr);
  EXPECT_EQ(as_impl(result)->size(), 0u);
}

TEST(IntelBufferBackendDmaBuf, CachedMappingIsNotReusedForLargerRequest)
{
  ExportedBuffer exported(0xd0b0004u);
  ASSERT_FALSE(exported.socket_path.empty());
  IntelBufferBackend backend;
  auto first = exported.descriptor(0xd0b0004u, kDmaBufBytes, kDmaBufBytes);
  auto ok = backend.from_descriptor_with_endpoint(&first, make_endpoint(43));
  ASSERT_NE(as_impl(ok)->data(), nullptr);

  // Same pool/block (served from the import cache), claiming a bigger buffer.
  auto bigger = exported.descriptor(0xd0b0004u, 4 * kDmaBufBytes, 4 * kDmaBufBytes);
  auto result = backend.from_descriptor_with_endpoint(&bigger, make_endpoint(44));
  EXPECT_EQ(as_impl(result)->data(), nullptr);
  EXPECT_EQ(as_impl(result)->size(), 0u);
}

// --- Refcounting is required for zero-copy ----------------------------------

namespace
{
// A same-host peer that imports DMA-BUF (and Level Zero, if advertised).
rmw_topic_endpoint_info_t discover_same_host_peer(
  IntelBufferBackend & backend, uint32_t gid_seed)
{
  auto endpoint = make_endpoint(gid_seed);
  std::unordered_map<std::string, std::string> supported;
  supported["intel_buffer"] = "transport=dmabuf|device=0|extmemmap=1" + local_host_entry(backend);
  EXPECT_TRUE(backend.on_discovering_endpoint(endpoint, {}, supported).first);
  return endpoint;
}
}  // namespace

// A block without refcount metadata could be reused while a subscriber reads
// it, so it is sent as a copy, not as a descriptor.
TEST(IntelBufferBackendDescriptor, BlockWithoutRefcountMetadataIsSentAsCopy)
{
  if (!IntelBufferImpl<uint8_t>::get_or_create_global_pool()) {
    GTEST_SKIP() << "no buffer pool on this machine";
  }
  IntelBufferBackend backend;
  auto peer = discover_same_host_peer(backend, 50);

  const int fd = memfd_create("ibb_no_meta", 0);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, kDmaBufBytes), 0);
  auto block = std::make_unique<intel_buffer_backend::PoolBlock>();
  block->ptr = mmap(nullptr, kDmaBufBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(block->ptr, MAP_FAILED);
  block->size = kDmaBufBytes;
  block->block_id = 0xd0b0005u;
  block->transport = intel_buffer_backend::TransportMode::DMA_BUF;
  block->dmabuf_fd = fd;   // exportable, but no ipc_meta
  IntelBufferImpl<uint8_t> impl(block.release(), kDmaBufBytes, /*owns_block=*/true,
    /*standalone=*/true);

  EXPECT_EQ(backend.create_descriptor_with_endpoint(&impl, peer), nullptr);
}

// Lifecycle of a wrapped camera dma-buf: the producer may only re-queue it once
// no live buffer wraps it and every issued descriptor was imported and
// released; a descriptor arriving after that is rejected.
TEST(IntelBufferBackendWrappedDmabuf, ReclaimWaitsForImportAndReleaseThenRejectsLateImports)
{
  if (!IntelBufferImpl<uint8_t>::get_or_create_global_pool()) {
    GTEST_SKIP() << "no buffer pool on this machine";
  }
  IntelBufferBackend backend;
  auto peer = discover_same_host_peer(backend, 51);
  const int fd = memfd_create("ibb_camera_frame", 0);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, kDmaBufBytes), 0);

  std::shared_ptr<IntelBufferDescriptor> desc;
  intel_memory_core::IPCMetadata * meta = nullptr;
  {
    auto frame = intel_buffer_backend::wrap_dmabuf(fd, kDmaBufBytes, /*owns_fd=*/false);
    auto * impl = static_cast<IntelBufferImpl<uint8_t> *>(frame.get_impl());
    ASSERT_NE(impl->get_block(), nullptr);
    meta = impl->get_block()->ipc_meta;
    ASSERT_NE(meta, nullptr);
    impl->data()[0] = 0x7a;
    desc = std::static_pointer_cast<IntelBufferDescriptor>(
      backend.create_descriptor_with_endpoint(impl, peer));
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->transport_mode, IntelBufferDescriptor::TRANSPORT_DMA_BUF);
    EXPECT_EQ(meta->pending_imports.load(), 1);
    EXPECT_FALSE(intel_buffer_backend::try_reclaim_dmabuf(fd));   // still wrapped
  }
  // Published, not yet imported by the subscriber.
  EXPECT_FALSE(intel_buffer_backend::try_reclaim_dmabuf(fd));

  {
    auto received = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(52));
    ASSERT_NE(as_impl(received)->data(), nullptr);
    EXPECT_EQ(as_impl(received)->data()[0], 0x7a);
    EXPECT_EQ(meta->pending_imports.load(), 0);
    EXPECT_FALSE(intel_buffer_backend::try_reclaim_dmabuf(fd));   // being read
  }
  EXPECT_TRUE(intel_buffer_backend::try_reclaim_dmabuf(fd));

  // The producer may now overwrite the buffer; the old descriptor is stale.
  auto late = backend.from_descriptor_with_endpoint(desc.get(), make_endpoint(53));
  EXPECT_EQ(as_impl(late)->data(), nullptr);
  close(fd);
}
