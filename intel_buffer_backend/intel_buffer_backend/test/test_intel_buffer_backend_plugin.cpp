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

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
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
    EXPECT_EQ(meta, "transport=extmemmap;extmemmap=1");
  } else {
    EXPECT_NE(meta.find("transport="), std::string::npos);
    EXPECT_NE(meta.find("device="), std::string::npos);
    EXPECT_NE(meta.find("extmemmap=1"), std::string::npos);
  }
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
  supported["intel_buffer"] = "transport=extmemmap;extmemmap=1";

  auto result = backend.on_discovering_endpoint(endpoint, {}, supported);
  EXPECT_TRUE(result.first);
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
  supported["intel_buffer"] = remote_transport;

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
  supported["intel_buffer"] = "transport=bogus;extmemmap=1";

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

TEST(IntelBufferBackendFromDescriptor, IntraProcessLevelZeroShortcutWrapsLocalPtrDirectly)
{
  const size_t size = 32;
  void * heap = std::malloc(size);
  std::memset(heap, 0x9, size);

  IntelBufferBackend backend;
  auto endpoint = make_endpoint(9);

  auto frame = std::make_shared<IntelBufferDescriptor>();
  frame->transport_mode = IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM;
  frame->l0_desc.size = size;
  frame->l0_desc.local_ptr = reinterpret_cast<uint64_t>(heap);
  frame->l0_desc.publisher_pid = static_cast<int32_t>(getpid());

  auto result = backend.from_descriptor_with_endpoint(frame.get(), endpoint);
  ASSERT_NE(result.get(), nullptr);
  auto * impl = static_cast<IntelBufferImpl<uint8_t> *>(result.get());
  EXPECT_EQ(impl->size(), size);
  EXPECT_EQ(impl->data(), heap);

  result.reset();
  std::free(heap);
}

TEST(IntelBufferBackendFromDescriptor, IntraProcessExternalMemmapShortcutWrapsLocalPtrDirectly)
{
  const size_t size = 16;
  uint8_t heap[size] = {};

  IntelBufferBackend backend;
  auto endpoint = make_endpoint(10);

  auto frame = std::make_shared<IntelBufferDescriptor>();
  frame->transport_mode = IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP;
  frame->extmap_desc.size = size;
  frame->extmap_desc.local_ptr = reinterpret_cast<uint64_t>(heap);
  frame->extmap_desc.publisher_pid = static_cast<int32_t>(getpid());

  auto result = backend.from_descriptor_with_endpoint(frame.get(), endpoint);
  ASSERT_NE(result.get(), nullptr);
  auto * impl = static_cast<IntelBufferImpl<uint8_t> *>(result.get());
  EXPECT_EQ(impl->size(), size);
  EXPECT_EQ(impl->data(), heap);
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
