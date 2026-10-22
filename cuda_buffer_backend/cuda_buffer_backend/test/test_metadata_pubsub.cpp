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

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <future>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer_backend/cuda_buffer_backend.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rmw_test_fixture/rmw_test_fixture.h"
#include "sensor_msgs/msg/image.hpp"

extern char ** environ;

namespace
{
int receive_in_child(bool matching)
{
  if (!matching) {
    const auto uid = std::to_string(geteuid() ^ 1u);
    setenv("CUDA_BUFFER_UID_OVERRIDE", uid.c_str(), 1);
  }
  rclcpp::init(0, nullptr);
  int result = 1;
  {
    auto node = std::make_shared<rclcpp::Node>("metadata_remote_subscriber");
    std::promise<void> complete;
    auto completed = complete.get_future();
    bool finished = false;
    size_t received = 0;
    rclcpp::SubscriptionOptions options;
    options.acceptable_buffer_backends = "cuda";
    auto subscription = node->create_subscription<sensor_msgs::msg::Image>(
      "metadata_remote", 10, [&](sensor_msgs::msg::Image::ConstSharedPtr image) {
        if (finished) {return;}
        if (image->data.to_vector() != std::vector<uint8_t>(32, 42)) {
          finished = true;
          complete.set_value();
          return;
        }
        const auto backend = image->data.get_backend_type();
        if (matching && backend != "cuda") {return;}
        if (!matching && backend != "cpu") {
          finished = true;
          complete.set_value();
          return;
        }
        if (++received == 10) {
          result = 0;
          finished = true;
          complete.set_value();
        }
      }, options);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    if (executor.spin_until_future_complete(completed, std::chrono::seconds(15)) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      result = 2;
    }
  }
  rclcpp::shutdown();
  return result;
}

void check_remote(bool matching)
{
  cuda_buffer_backend::CudaBufferBackend backend;
  const auto metadata =
    cuda_buffer_backend::detail::parse_ipc_metadata(backend.get_backend_metadata());
  ASSERT_TRUE(metadata);
  if (!metadata->capable) {GTEST_SKIP() << "Requires IPC-capable CUDA";}
  auto node = std::make_shared<rclcpp::Node>("metadata_remote_publisher");
  auto publisher = node->create_publisher<sensor_msgs::msg::Image>("metadata_remote", 10);
  sensor_msgs::msg::Image image;
  image.data = cuda_buffer_backend::allocate_buffer(32);
  {
    auto handle = cuda_buffer_backend::from_output_buffer(image.data, nullptr);
    ASSERT_EQ(cudaMemset(handle.get_ptr(), 42, 32), cudaSuccess);
  }
  auto timer = node->create_wall_timer(std::chrono::milliseconds(20), [&]() {
        publisher->publish(image);
  });
  pid_t child;
  char * arguments[] = {const_cast<char *>("/proc/self/exe"),
    const_cast<char *>("--metadata-peer"), const_cast<char *>(matching ? "cuda" : "cpu"), nullptr};
  ASSERT_EQ(posix_spawn(&child, arguments[0], nullptr, nullptr, arguments, environ), 0);
  // The child has its own bounded executor timeout and is always reaped.
  auto exited = std::async(std::launch::async, [child]() {
        int status = 0;
        if (waitpid(child, &status, 0) != child || !WIFEXITED(status)) {return -1;}
        return WEXITSTATUS(status);
  });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  EXPECT_EQ(executor.spin_until_future_complete(exited, std::chrono::seconds(20)),
    rclcpp::FutureReturnCode::SUCCESS);
  EXPECT_EQ(exited.get(), 0);
}
}  // namespace

TEST(MetadataPubSub, SeparateProcessCudaDelivery)
{
  check_remote(true);
}

TEST(MetadataPubSub, SeparateProcessUidMismatchUsesHost)
{
  check_remote(false);
}

TEST(MetadataPubSub, CudaAndHostDelivery)
{
  cuda_buffer_backend::CudaBufferBackend backend;
  const auto metadata =
    cuda_buffer_backend::detail::parse_ipc_metadata(backend.get_backend_metadata());
  ASSERT_TRUE(metadata);
  if (!metadata->capable) {GTEST_SKIP() << "Requires IPC-capable CUDA";}
  auto node = std::make_shared<rclcpp::Node>("metadata_cuda_test");
  auto publisher = node->create_publisher<sensor_msgs::msg::Image>("metadata_image", 10);
  bool cuda_received = false;
  bool cpu_received = false;
  bool finished = false;
  std::promise<void> complete;
  auto completed = complete.get_future();
  auto callback = [&](const sensor_msgs::msg::Image & image, bool expect_cuda) {
      EXPECT_EQ(image.data.to_vector(), std::vector<uint8_t>(32, 42));
      if (expect_cuda) {
        // Initial host fallback before discovery is allowed, but CUDA must follow.
        if (image.data.get_backend_type() == "cuda") {cuda_received = true;}
      } else {
        EXPECT_EQ(image.data.get_backend_type(), "cpu");
        cpu_received = true;
      }
      if (cpu_received && cuda_received && !finished) {
        finished = true;
        complete.set_value();
      }
    };
  rclcpp::SubscriptionOptions options;
  options.acceptable_buffer_backends = "cuda";
  auto cuda_sub = node->create_subscription<sensor_msgs::msg::Image>("metadata_image", 10,
      [&](sensor_msgs::msg::Image::ConstSharedPtr image) {callback(*image, true);}, options);
  options.acceptable_buffer_backends = "cpu";
  auto cpu_sub = node->create_subscription<sensor_msgs::msg::Image>("metadata_image", 10,
      [&](sensor_msgs::msg::Image::ConstSharedPtr image) {callback(*image, false);}, options);
  sensor_msgs::msg::Image image;
  image.data = cuda_buffer_backend::allocate_buffer(32);
  {
    auto handle = cuda_buffer_backend::from_output_buffer(image.data, nullptr);
    ASSERT_EQ(cudaMemset(handle.get_ptr(), 42, 32), cudaSuccess);
  }
  auto timer = node->create_wall_timer(std::chrono::milliseconds(20), [&]() {
        publisher->publish(image);
  });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  EXPECT_EQ(executor.spin_until_future_complete(completed, std::chrono::seconds(20)),
    rclcpp::FutureReturnCode::SUCCESS);
}

int main(int argc, char ** argv)
{
  if (argc == 3 && std::string(argv[1]) == "--metadata-peer") {
    return receive_in_child(std::string(argv[2]) == "cuda");
  }
  testing::InitGoogleTest(&argc, argv);
  if (rmw_test_isolation_start() != RMW_RET_OK) {return 1;}
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  if (rmw_test_isolation_stop() != RMW_RET_OK) {return 1;}
  return result;
}
