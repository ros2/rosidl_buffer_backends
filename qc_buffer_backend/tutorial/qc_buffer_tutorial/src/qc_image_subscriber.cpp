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

// Minimal subscriber demonstrating qc_buffer_backend.
//
// It accepts any buffer backend. When the message arrives qc-backed (same
// process as the publisher), it reads the ION/dma-buf memory directly through
// the CPU pointer and can hand the dma-buf fd to the HTP for zero-copy access.
// Otherwise it reads the payload the normal way (CPU fallback).

#include <memory>

#include "qc_buffer/qc_buffer_api.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"

class QcImageSubscriber : public rclcpp::Node
{
public:
  explicit QcImageSubscriber(const rclcpp::NodeOptions & options)
  : Node("qc_image_subscriber", options), count_(0)
  {
    // Accept whatever backend the publisher used (qc when co-located, cpu
    // otherwise).
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.acceptable_buffer_backends = "any";

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "qc_image", 10,
      std::bind(&QcImageSubscriber::on_image, this, std::placeholders::_1),
      sub_opts);

    RCLCPP_INFO(this->get_logger(), "qc_image_subscriber started");
  }

private:
  void on_image(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    ++count_;
    const std::string backend = msg->data.get_backend_type();

    if (backend == "qc") {
      // Zero-copy: read the same ION memory the publisher wrote.
      const int fd = qc_buffer_backend::get_dmabuf_fd(msg->data);
      uint8_t * p = qc_buffer_backend::get_data_ptr(msg->data);
      const uint8_t first = (p != nullptr && msg->data.size() > 0) ? p[0] : 0;

      RCLCPP_INFO(this->get_logger(),
        "image #%zu: %ux%u, %zu bytes, backend=qc, dmabuf_fd=%d, first_byte=%u "
        "(pass fd to the HTP accelerator for zero-copy)",
        count_, msg->width, msg->height, msg->data.size(), fd, first);
    } else {
      // CPU fallback: read the payload the normal way.
      const std::vector<uint8_t> & data = msg->data;
      const uint8_t first = data.empty() ? 0 : data[0];
      RCLCPP_INFO(this->get_logger(),
        "image #%zu: %ux%u, %zu bytes, backend=%s (CPU fallback), first_byte=%u",
        count_, msg->width, msg->height, msg->data.size(), backend.c_str(), first);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  size_t count_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(QcImageSubscriber)
