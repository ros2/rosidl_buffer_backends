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

#include <hip/hip_runtime_api.h>

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "hip_buffer/hip_buffer_api.hpp"

class HipImageGpuRelay : public rclcpp::Node
{
public:
  explicit HipImageGpuRelay(const rclcpp::NodeOptions & options)
  : Node("hip_image_gpu_relay", options), relay_count_(0)
  {
    this->declare_parameter<std::string>("input_topic", "test_hip_image");
    this->declare_parameter<std::string>("output_topic", "test_hip_image_relayed");

    std::string input_topic = this->get_parameter("input_topic").as_string();
    std::string output_topic = this->get_parameter("output_topic").as_string();

    hipStreamCreate(&stream_);

    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.acceptable_buffer_backends = "any";
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      input_topic, 10,
      std::bind(&HipImageGpuRelay::image_callback, this, std::placeholders::_1),
      sub_opts);

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>(output_topic, 10);
    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>("relay_count", 10);

    RCLCPP_INFO(this->get_logger(), "GPU relay node started (no CPU sync)");
  }

  ~HipImageGpuRelay() override
  {
    if (stream_ != nullptr) {
      hipStreamSynchronize(stream_);
      hipStreamDestroy(stream_);
    }
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const std::string backend = msg->data.get_backend_type();
    if (backend != "hip") {
      RCLCPP_WARN(this->get_logger(), "Skipping non-HIP message (backend: %s)",
        backend.c_str());
      return;
    }

    size_t byte_size = msg->data.size();

    sensor_msgs::msg::Image out;
    out.data = hip_buffer_backend::allocate_buffer(byte_size);
    out.header = msg->header;
    out.height = msg->height;
    out.width = msg->width;
    out.encoding = msg->encoding;
    out.step = msg->step;
    out.is_bigendian = msg->is_bigendian;

    // Event-ordered D2D relay; intentionally no host sync.
    {
      const rosidl::Buffer<uint8_t> & src = msg->data;
      hip_buffer_backend::ReadHandle rh =
        hip_buffer_backend::from_input_buffer(src, stream_);
      hip_buffer_backend::WriteHandle wh =
        hip_buffer_backend::from_output_buffer(out.data, stream_);
      hipMemcpyAsync(wh.get_ptr(), rh.get_ptr(), byte_size,
        hipMemcpyDeviceToDevice, stream_);
    }

    publisher_->publish(out);

    std_msgs::msg::UInt32 count_msg;
    count_msg.data = ++relay_count_;
    count_publisher_->publish(count_msg);

    if (relay_count_ % 10 == 0) {
      RCLCPP_INFO(this->get_logger(), "Relayed %u images (GPU-only, no CPU sync)",
        relay_count_);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  hipStream_t stream_{nullptr};
  uint32_t relay_count_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(HipImageGpuRelay)
