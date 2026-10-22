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
#include <memory>
#include <set>

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

TEST_F(CudaBufferBackendTest, UndiscoveredPeerUsesHostFallback)
{
  rmw_topic_endpoint_info_t endpoint_info{};
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(backend_->create_descriptor_with_endpoint(nullptr, endpoint_info), nullptr);
  }
}

TEST_F(CudaBufferBackendTest, CapabilityProbeWithoutUsableDevice)
{
  int count = 0;
  if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
    GTEST_SKIP() << "Run with CUDA_VISIBLE_DEVICES empty to exercise this path";
  }
  EXPECT_NO_THROW(EXPECT_FALSE(CudaBufferImpl<uint8_t>::is_pool_ipc_capable()));
  rmw_topic_endpoint_info_t endpoint_info{};
  EXPECT_FALSE(backend_->on_discovering_endpoint(endpoint_info, {}, {{"cuda", ""}}).first);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
