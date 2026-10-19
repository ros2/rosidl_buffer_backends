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

// Minimal publisher demonstrating qc_buffer_backend.
//
// It allocates a qc-backed rosidl::Buffer<uint8_t>, fills it directly through
// the CPU pointer (no copy), and publishes it. When a subscriber lives in the
// same process, delivery is zero-copy.

#include <chrono>
#include <memory>

#include "qc_buffer/qc_buffer_api.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"

class QcImagePublisher : public rclcpp::Node
{
public:
  explicit QcImagePublisher(const rclcpp::NodeOptions & options)
  : Node("qc_image_publisher", options), count_(0)
  {
    width_ = this->declare_parameter<int>("image_width", 8);
    height_ = this->declare_parameter<int>("image_height", 8);

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("qc_image", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&QcImagePublisher::tick, this));

    RCLCPP_INFO(this->get_logger(), "qc_image_publisher started (%dx%d)", width_, height_);
  }

private:
  void tick()
  {
    const size_t size = static_cast<size_t>(width_) * height_ * 3;

    sensor_msgs::msg::Image msg;
    // Allocate a qc-backed buffer for the payload.
    msg.data = qc_buffer_backend::allocate_buffer(size);
    msg.header.stamp = this->now();
    msg.header.frame_id = "qc_frame";
    msg.height = height_;
    msg.width = width_;
    msg.encoding = "rgb8";
    msg.step = width_ * 3;

    // Write directly to the ION/dma-buf memory through the CPU pointer.
    uint8_t * p = qc_buffer_backend::get_data_ptr(msg.data);
    if (p != nullptr) {
      std::fill(p, p + size, static_cast<uint8_t>(count_ % 256));
    } else {
      // Not qc-backed (e.g. rpcmem unavailable): fall back to normal access.
      for (size_t i = 0; i < size; ++i) {
        msg.data[i] = static_cast<uint8_t>(count_ % 256);
      }
    }

    publisher_->publish(msg);

    if (++count_ % 10 == 0) {
      RCLCPP_INFO(this->get_logger(), "published %zu images (backend: %s, fd: %d)",
        count_, msg.data.get_backend_type().c_str(),
        qc_buffer_backend::get_dmabuf_fd(msg.data));
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  size_t count_;
  int width_;
  int height_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(QcImagePublisher)
