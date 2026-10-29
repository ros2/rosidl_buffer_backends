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
#include <memory>
#include <set>

#include "hip_buffer/hip_buffer_impl.hpp"
#include "hip_buffer_backend/hip_buffer_backend.hpp"
#include "hip_buffer_backend_msgs/msg/hip_buffer_descriptor.hpp"

using hip_buffer_backend::HipBufferBackend;
using hip_buffer_backend::HipBufferImpl;

class HipBufferBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    backend_ = std::make_unique<HipBufferBackend>();
  }
  void TearDown() override {}

  std::unique_ptr<HipBufferBackend> backend_;
};

TEST_F(HipBufferBackendTest, BackendTypeName)
{
  EXPECT_EQ(backend_->get_backend_type(), "hip");
}

TEST_F(HipBufferBackendTest, OnDiscoveringEndpointWithoutHipSupport)
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

// Ported from ros2/rosidl_buffer_backends PR #26. An unknown subscriber has
// no local IPC decision, so HIP descriptor creation must select the host path.
TEST_F(HipBufferBackendTest, NoDescriptorForUnknownSubscriber)
{
  rmw_topic_endpoint_info_t endpoint_info;
  std::memset(&endpoint_info, 0, sizeof(endpoint_info));
  endpoint_info.topic_type = "test_type";
  endpoint_info.endpoint_type = RMW_ENDPOINT_SUBSCRIPTION;
  std::memset(endpoint_info.endpoint_gid, 0x5a, sizeof(endpoint_info.endpoint_gid));

  std::vector<rmw_topic_endpoint_info_t> existing_endpoints;
  std::unordered_map<std::string, std::string> supported_backends;
  supported_backends["hip"] = "";
  auto result = backend_->on_discovering_endpoint(
    endpoint_info, existing_endpoints, supported_backends);
  EXPECT_FALSE(result.first);

  HipBufferImpl<uint8_t> impl(1024);
  EXPECT_EQ(nullptr, backend_->create_descriptor_with_endpoint(&impl, endpoint_info));
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
