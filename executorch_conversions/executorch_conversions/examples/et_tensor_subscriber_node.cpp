// Copyright 2026 Open Source Robotics Foundation, Inc.
// Copyright 2026 Meta Platforms, Inc. and affiliates.
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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "executorch_conversions/executorch_conversions.hpp"

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;

namespace
{

template<typename T>
std::pair<double, double> min_max(const void * data, std::size_t count)
{
  const auto * values = static_cast<const T *>(data);
  const auto [minimum, maximum] = std::minmax_element(values, values + count);
  return {static_cast<double>(*minimum), static_cast<double>(*maximum)};
}

std::pair<double, double> min_max(ScalarType type, const void * data, std::size_t count)
{
  switch (type) {
    case ScalarType::Byte: return min_max<uint8_t>(data, count);
    case ScalarType::Char: return min_max<int8_t>(data, count);
    case ScalarType::Short: return min_max<int16_t>(data, count);
    case ScalarType::Int: return min_max<int32_t>(data, count);
    case ScalarType::Long: return min_max<int64_t>(data, count);
    case ScalarType::Half: return min_max<executorch::aten::Half>(data, count);
    case ScalarType::BFloat16: return min_max<executorch::aten::BFloat16>(data, count);
    case ScalarType::Float: return min_max<float>(data, count);
    case ScalarType::Double: return min_max<double>(data, count);
    // Any byte can arrive, and only 0 and 1 are valid C++ bools.
    case ScalarType::Bool: return min_max<uint8_t>(data, count);
    default: throw std::invalid_argument("unsupported dtype");
  }
}

}  // namespace

class ExecuTorchTensorSubscriber : public rclcpp::Node
{
public:
  explicit ExecuTorchTensorSubscriber(const rclcpp::NodeOptions & options)
  : Node("et_tensor_subscriber", options)
  {
    rclcpp::SubscriptionOptions subscription_options;
    // Without the CUDA plugin this node cannot read a CUDA message, so it takes the host copy.
    subscription_options.acceptable_buffer_backends =
      executorch_conversions::backend_available("cuda") ? "any" : "cpu";
    subscription_ = create_subscription<tensor_msgs::msg::ExperimentalTensor>(
      "test_et_tensor", 10,
      [this](tensor_msgs::msg::ExperimentalTensor::ConstSharedPtr received) {
        receive_tensor(*received);
      }, subscription_options);
  }

private:
  void receive_tensor(const tensor_msgs::msg::ExperimentalTensor & received)
  {
    // Any publisher can send a message the library cannot view, so skip it.
    try {
      // Borrow the message storage; the message stays alive throughout this callback.
      auto input_tensor = executorch_conversions::from_input_tensor_msg(received);
      if (!input_tensor || input_tensor->numel() == 0) {
        return;
      }

      // Replace this with your own processing, for example Module::execute on
      // the view. Here the values are copied to a host message only to print them.
      const auto type = input_tensor->scalar_type();
      std::vector<int64_t> shape(input_tensor->sizes().begin(), input_tensor->sizes().end());
      auto host = executorch_conversions::allocate_tensor_msg(
        shape, type, Device(DeviceType::CPU));
      executorch_conversions::to_tensor_msg(*host, *input_tensor);
      const auto [minimum, maximum] = min_max(
        type, host->data.data(), static_cast<std::size_t>(input_tensor->numel()));
      RCLCPP_INFO(
        get_logger(), "Received tensor (backend=%s, min=%g, max=%g, count=%zu)",
        received.data.get_backend_type().c_str(), minimum, maximum, ++received_count_);
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "Skipping a message: %s", error.what());
    }
  }

  rclcpp::Subscription<tensor_msgs::msg::ExperimentalTensor>::SharedPtr subscription_;
  std::size_t received_count_{0};
};

RCLCPP_COMPONENTS_REGISTER_NODE(ExecuTorchTensorSubscriber)
