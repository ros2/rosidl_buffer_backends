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

#include <gtest/gtest.h>

#include <unistd.h>

#include <memory>
#include <string>
#include <typeinfo>

#include "qc_buffer/fd_broker.hpp"
#include "qc_buffer/qc_buffer.hpp"
#include "qc_buffer/qc_buffer_api.hpp"
#include "qc_buffer/qc_buffer_impl.hpp"
#include "qc_buffer/rpcmem_loader.hpp"
#include "qc_buffer_backend/qc_buffer_backend.hpp"
#include "qc_buffer_backend_msgs/msg/qc_buffer_descriptor.hpp"
#include "rmw/topic_endpoint_info.h"

using qc_buffer_backend::FdBroker;
using qc_buffer_backend::QcBuffer;
using qc_buffer_backend::QcBufferBackend;
using qc_buffer_backend::QcBufferImpl;
using qc_buffer_backend::RpcMemLoader;

namespace
{

/// Build a zeroed endpoint info (no GID information).
rmw_topic_endpoint_info_t make_endpoint()
{
  rmw_topic_endpoint_info_t ep = rmw_get_zero_initialized_topic_endpoint_info();
  return ep;
}

}  // namespace

// ── QcBufferBackend basic interface ──────────────────────────────────────────

// Backend type string is "qc".
TEST(QcBufferBackendTest, BackendTypeIsQc)
{
  QcBufferBackend backend;
  EXPECT_EQ(backend.get_backend_type(), "qc");
}

// Descriptor type support is non-null.
TEST(QcBufferBackendTest, DescriptorTypeSupportNonNull)
{
  QcBufferBackend backend;
  EXPECT_NE(backend.get_descriptor_type_support(), nullptr);
}

// create_empty_descriptor returns a non-null, zero-initialized descriptor.
TEST(QcBufferBackendTest, CreateEmptyDescriptorNonNull)
{
  QcBufferBackend backend;
  auto desc = backend.create_empty_descriptor();
  ASSERT_NE(desc, nullptr);
  const auto * typed =
    static_cast<const qc_buffer_backend_msgs::msg::QcBufferDescriptor *>(desc.get());
  EXPECT_EQ(typed->uid, 0u);
  EXPECT_FALSE(typed->use_ipc);
}

// ── create_descriptor_with_endpoint ──────────────────────────────────────────

// Returns nullptr for a non-QcBufferImpl impl pointer.
TEST(QcBufferBackendTest, CreateDescriptorReturnsNullForNonQcImpl)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBufferBackend backend;
  auto ep = make_endpoint();
  // nullptr impl must return nullptr immediately.
  auto desc = backend.create_descriptor_with_endpoint(nullptr, ep);
  EXPECT_EQ(desc, nullptr);
}

// Successful path: descriptor is populated with correct fields.
TEST(QcBufferBackendTest, CreateDescriptorPopulatesFields)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBufferBackend backend;
  auto ep = make_endpoint();

  QcBufferImpl<uint8_t> impl(32);
  auto desc_ptr = backend.create_descriptor_with_endpoint(&impl, ep);
  ASSERT_NE(desc_ptr, nullptr);

  const auto * desc =
    static_cast<const qc_buffer_backend_msgs::msg::QcBufferDescriptor *>(desc_ptr.get());

  EXPECT_EQ(desc->size, 32u);
  EXPECT_EQ(desc->element_type_name, std::string(typeid(uint8_t).name()));
  EXPECT_EQ(desc->pid, static_cast<int32_t>(getpid()));
  EXPECT_GT(desc->uid, 0u);
  EXPECT_GE(desc->dmabuf_fd, 0);
  EXPECT_GT(desc->dmabuf_size, 0u);
  EXPECT_TRUE(desc->use_ipc);
  EXPECT_FALSE(desc->ipc_socket_path.empty());
}

// Descriptor ipc_socket_path contains the current pid.
TEST(QcBufferBackendTest, CreateDescriptorSocketPathContainsPid)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBufferBackend backend;
  auto ep = make_endpoint();

  QcBufferImpl<uint8_t> impl(16);
  auto desc_ptr = backend.create_descriptor_with_endpoint(&impl, ep);
  ASSERT_NE(desc_ptr, nullptr);

  const auto * desc =
    static_cast<const qc_buffer_backend_msgs::msg::QcBufferDescriptor *>(desc_ptr.get());
  EXPECT_NE(
    desc->ipc_socket_path.find(std::to_string(getpid())),
    std::string::npos);
}

// ── from_descriptor_with_endpoint ────────────────────────────────────────────

// Returns empty impl (not null) for descriptor with mismatched element type.
TEST(QcBufferBackendTest, FromDescriptorReturnsEmptyOnTypeMismatch)
{
  QcBufferBackend backend;
  auto ep = make_endpoint();

  qc_buffer_backend_msgs::msg::QcBufferDescriptor desc;
  desc.element_type_name = "wrong_type";
  desc.use_ipc = false;

  auto result = backend.from_descriptor_with_endpoint(&desc, ep);
  ASSERT_NE(result.get(), nullptr);
  const auto * impl =
    static_cast<const rosidl::BufferImplBase<uint8_t> *>(result.get());
  EXPECT_EQ(impl->size(), 0u);
}

// Returns empty impl for descriptor without IPC socket path.
TEST(QcBufferBackendTest, FromDescriptorReturnsEmptyWhenNoSocketPath)
{
  QcBufferBackend backend;
  auto ep = make_endpoint();

  qc_buffer_backend_msgs::msg::QcBufferDescriptor desc;
  desc.element_type_name = typeid(uint8_t).name();
  desc.use_ipc = false;
  desc.ipc_socket_path = "";

  auto result = backend.from_descriptor_with_endpoint(&desc, ep);
  ASSERT_NE(result.get(), nullptr);
  const auto * impl =
    static_cast<const rosidl::BufferImplBase<uint8_t> *>(result.get());
  EXPECT_EQ(impl->size(), 0u);
}

// Full roundtrip: create descriptor then reconstruct impl from it.
TEST(QcBufferBackendTest, RoundTripZeroCopy)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBufferBackend backend;
  auto ep = make_endpoint();

  // Publisher side: allocate and write a known pattern.
  QcBufferImpl<uint8_t> pub_impl(8);
  pub_impl.get_qc_buffer()->data()[0] = 0xDE;
  pub_impl.get_qc_buffer()->data()[7] = 0xAD;

  auto desc_ptr = backend.create_descriptor_with_endpoint(&pub_impl, ep);
  ASSERT_NE(desc_ptr, nullptr);

  // Subscriber side: reconstruct from descriptor.
  auto sub_result = backend.from_descriptor_with_endpoint(desc_ptr.get(), ep);
  ASSERT_NE(sub_result.get(), nullptr);

  const auto * sub_impl =
    static_cast<const QcBufferImpl<uint8_t> *>(sub_result.get());
  ASSERT_NE(sub_impl->get_qc_buffer(), nullptr);

  EXPECT_EQ(sub_impl->size(), 8u);
  EXPECT_EQ(sub_impl->get_backend_type(), "qc");
  // Same physical memory: subscriber sees the bytes the publisher wrote.
  EXPECT_EQ(sub_impl->get_qc_buffer()->data()[0], 0xDE);
  EXPECT_EQ(sub_impl->get_qc_buffer()->data()[7], 0xAD);
  // dma-buf fd is available for HTP accelerator registration.
  EXPECT_GE(sub_impl->get_qc_buffer()->dmabuf_fd(), 0);
}

// Roundtrip with broker miss (stale uid) returns empty impl, not crash.
TEST(QcBufferBackendTest, FromDescriptorReturnsFallbackOnBrokerMiss)
{
  QcBufferBackend backend;
  auto ep = make_endpoint();

  qc_buffer_backend_msgs::msg::QcBufferDescriptor desc;
  desc.element_type_name = typeid(uint8_t).name();
  desc.use_ipc = true;
  desc.uid = 0xDEADBEEF0ULL;
  desc.dmabuf_size = 64;
  desc.size = 64;
  // Use the current broker socket (broker is running but uid is unknown).
  desc.ipc_socket_path = FdBroker::instance().socket_path();

  auto result = backend.from_descriptor_with_endpoint(&desc, ep);
  ASSERT_NE(result.get(), nullptr);
  const auto * impl =
    static_cast<const rosidl::BufferImplBase<uint8_t> *>(result.get());
  EXPECT_EQ(impl->size(), 0u);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
