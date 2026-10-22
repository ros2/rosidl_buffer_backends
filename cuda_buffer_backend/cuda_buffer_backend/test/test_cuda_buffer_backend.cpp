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
#include <future>
#include <memory>
#include <set>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer_backend/ipc_metadata.hpp"

#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer_backend/cuda_buffer_backend.hpp"
#include "cuda_buffer_backend_msgs/msg/cuda_buffer_descriptor.hpp"

using cuda_buffer_backend::CudaBufferBackend;
using cuda_buffer_backend::CudaBufferImpl;

class CudaBufferBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    backend_ = std::make_unique<CudaBufferBackend>();
  }
  void TearDown() override {}

  std::unique_ptr<CudaBufferBackend> backend_;
};

TEST_F(CudaBufferBackendTest, BackendTypeName)
{
  EXPECT_EQ(backend_->get_backend_type(), "cuda");
}

TEST_F(CudaBufferBackendTest, OnDiscoveringEndpointWithoutCudaSupport)
{
  rmw_topic_endpoint_info_t endpoint_info;
  std::memset(&endpoint_info, 0, sizeof(endpoint_info));
  endpoint_info.topic_type = "test_type";

  std::vector<rmw_topic_endpoint_info_t> existing_endpoints;
  std::unordered_map<std::string, std::string> supported_backends;
  supported_backends["demo"] = "version=1.0";

  std::pair<bool, std::vector<std::set<uint32_t>>> result =
    backend_->on_discovering_endpoint(
      endpoint_info, existing_endpoints, supported_backends);
  EXPECT_FALSE(result.first);
}

TEST_F(CudaBufferBackendTest, MetadataAvailableBeforeEndpointCreationAndStableAcrossThreads)
{
  const auto advertised = backend_->get_backend_metadata();
  ASSERT_TRUE(cuda_buffer_backend::detail::parse_ipc_metadata(advertised));
  std::vector<std::future<std::string>> reads;
  for (int i = 0; i < 8; ++i) {
    reads.push_back(std::async(std::launch::async, [&]() {
        return backend_->get_backend_metadata();
      }));
  }
  for (auto & read : reads) {
    EXPECT_EQ(read.get(), advertised);
                                                               }
  CudaBufferBackend other;
  EXPECT_EQ(other.get_backend_metadata(), advertised);
}

TEST_F(CudaBufferBackendTest, MissingOrInvalidAdvertisementAndUnknownPeerFallBack)
{
  rmw_topic_endpoint_info_t peer{};
  for (const auto & metadata : {"", "garbage", "cuda-ipc-v1|0", "cuda-ipc-v2|1"}) {
    EXPECT_FALSE(backend_->on_discovering_endpoint(peer, {}, {{"cuda", metadata}}).first);
    EXPECT_EQ(backend_->create_descriptor_with_endpoint(nullptr, peer), nullptr);
  }
  peer.endpoint_gid[0] = 1;
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(backend_->create_descriptor_with_endpoint(nullptr, peer), nullptr);
  }
}

TEST_F(CudaBufferBackendTest, NoDeviceAdvertisesUnavailableWithoutThrowing)
{
  int count = 0;
  if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
    GTEST_SKIP() << "Run with CUDA_VISIBLE_DEVICES empty";
  }
  EXPECT_NO_THROW(EXPECT_FALSE(CudaBufferImpl<uint8_t>::is_pool_ipc_capable()));
  EXPECT_EQ(backend_->get_backend_metadata(), "cuda-ipc-v1|0");
}

TEST_F(CudaBufferBackendTest, DiscoveryAloneEnablesDescriptorWithoutHostRegistration)
{
  using cuda_buffer_backend::detail::parse_ipc_metadata;
  using cuda_buffer_backend::detail::serialize_ipc_metadata;
  const auto local = parse_ipc_metadata(backend_->get_backend_metadata());
  ASSERT_TRUE(local);
  if (!local->capable) {GTEST_SKIP() << "Requires IPC-capable CUDA";}
  rmw_topic_endpoint_info_t peer{};
  peer.endpoint_gid[0] = 42;
  auto buffer = cuda_buffer_backend::allocate_buffer(32);
  {
    auto handle = cuda_buffer_backend::from_output_buffer(buffer, nullptr);
    ASSERT_EQ(cudaMemset(handle.get_ptr(), 42, 32), cudaSuccess);
  }
  EXPECT_EQ(backend_->create_descriptor_with_endpoint(buffer.get_impl(), peer), nullptr);
  ASSERT_TRUE(backend_->on_discovering_endpoint(
    peer, {}, {{"cuda", backend_->get_backend_metadata()}}).first);
  EXPECT_NE(backend_->create_descriptor_with_endpoint(buffer.get_impl(), peer), nullptr);

  auto wrong_gpu = *local;
  wrong_gpu.gpu[0] = wrong_gpu.gpu[0] == '0' ? '1' : '0';
  ASSERT_FALSE(backend_->on_discovering_endpoint(
    peer, {}, {{"cuda", serialize_ipc_metadata(wrong_gpu)}}).first);
  EXPECT_EQ(backend_->create_descriptor_with_endpoint(buffer.get_impl(), peer), nullptr);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
