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

#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer/host_endpoint_manager.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace
{
struct DeferredRegistration
{
  host_endpoint_manager::HostEndpointManager * manager;
  rmw_gid_t gid;
  std::string topic;
};
std::mutex registration_mutex;
bool defer_registration = true;
std::vector<DeferredRegistration> deferred;

size_t release_registrations()
{
  std::vector<DeferredRegistration> registrations;
  {
    std::lock_guard<std::mutex> lock(registration_mutex);
    defer_registration = false;
    registrations.swap(deferred);
  }
  for (const auto & registration : registrations) {
    EXPECT_TRUE(registration.manager->register_subscription(
      registration.gid, registration.topic.c_str()));
  }
  return registrations.size();
}
}  // namespace

// Test-only ELF interposition: postpone registry writes without changing either
// RMW's endpoint ordering or production code. Export this symbol from the test
// executable so the dynamically loaded CUDA plugin uses this implementation.
namespace host_endpoint_manager
{
bool HostEndpointManager::register_subscription(const rmw_gid_t & gid, const char * topic)
{
  {
    std::lock_guard<std::mutex> lock(registration_mutex);
    if (defer_registration) {
      deferred.push_back({this, gid, topic ? topic : ""});
      return true;
    }
  }
  return register_endpoint(gid, topic, EntityType::SUBSCRIPTION);
}
}  // namespace host_endpoint_manager

TEST(DelayedRegistration, HostDeliveryThenCudaOnTheSameChannel)
{
  if (!cuda_buffer_backend::CudaBufferImpl<uint8_t>::is_pool_ipc_capable()) {
    GTEST_SKIP() << "Requires an IPC-capable CUDA device";
  }
  auto node = std::make_shared<rclcpp::Node>("delayed_cuda_registration");
  auto publisher = node->create_publisher<sensor_msgs::msg::Image>("delayed_cuda", 10);
  bool host_received = false;
  bool cuda_received = false;
  std::promise<void> complete;
  auto completed = complete.get_future();
  rclcpp::SubscriptionOptions options;
  options.acceptable_buffer_backends = "cuda";
  auto subscription = node->create_subscription<sensor_msgs::msg::Image>(
    "delayed_cuda", 10, [&](sensor_msgs::msg::Image::ConstSharedPtr message) {
      const auto bytes = message->data.to_vector();
      EXPECT_EQ(bytes, std::vector<uint8_t>(32, 42));
      if (message->data.get_backend_type() == "cpu" && !host_received) {
        host_received = true;
        // Delivery is the barrier: registry registration cannot happen until
        // the established endpoint channel has successfully delivered host data.
        EXPECT_GT(release_registrations(), 0u);
      } else if (message->data.get_backend_type() == "cuda" && !cuda_received) {
        EXPECT_TRUE(host_received);
        cuda_received = true;
        complete.set_value();
      }
    }, options);
  sensor_msgs::msg::Image message;
  message.data = cuda_buffer_backend::allocate_buffer(32);
  {
    auto handle = cuda_buffer_backend::from_output_buffer(message.data, nullptr);
    ASSERT_EQ(cudaMemset(handle.get_ptr(), 42, 32), cudaSuccess);
  }
  auto timer = node->create_wall_timer(std::chrono::milliseconds(20), [&]() {
        publisher->publish(message);
  });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  EXPECT_EQ(executor.spin_until_future_complete(completed, std::chrono::seconds(20)),
    rclcpp::FutureReturnCode::SUCCESS);
  EXPECT_TRUE(host_received);
  EXPECT_TRUE(cuda_received);
  release_registrations();
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
