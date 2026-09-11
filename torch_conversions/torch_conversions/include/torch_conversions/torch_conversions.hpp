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

#ifndef TORCH_CONVERSIONS__TORCH_CONVERSIONS_HPP_
#define TORCH_CONVERSIONS__TORCH_CONVERSIONS_HPP_

#include <torch/torch.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensor_msgs/msg/experimental_tensor.hpp"
#include "torch_conversions/visibility_control.hpp"

namespace torch_conversions
{

using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

TORCH_CONVERSIONS_PUBLIC std::vector<std::string> available_backends();

TORCH_CONVERSIONS_PUBLIC bool backend_available(
  const std::string & backend);

TORCH_CONVERSIONS_PUBLIC std::string backend_for_device(
  c10::DeviceType device_type);

TORCH_CONVERSIONS_PUBLIC std::string default_backend();

TORCH_CONVERSIONS_PUBLIC at::ScalarType scalar_type(
  const TensorMsg & msg);

TORCH_CONVERSIONS_PUBLIC std::vector<int64_t> normalized_strides(
  const TensorMsg & msg);

TORCH_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::DeviceType> device = std::nullopt);

/// CPU views borrow msg storage. Release output views before publishing msg.
TORCH_CONVERSIONS_PUBLIC at::Tensor from_output_tensor_msg(
  TensorMsg & msg,
  void * execution_stream = nullptr);

/// With clone=false, CPU storage must remain alive and unchanged in size.
TORCH_CONVERSIONS_PUBLIC at::Tensor from_input_tensor_msg(
  const TensorMsg & msg,
  bool clone = true,
  void * execution_stream = nullptr);

TORCH_CONVERSIONS_PUBLIC void to_tensor_msg(
  TensorMsg & msg,
  const at::Tensor & tensor,
  void * execution_stream = nullptr);

TORCH_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> to_tensor_msg(
  const at::Tensor & tensor,
  void * execution_stream = nullptr);

}  // namespace torch_conversions

#endif  // TORCH_CONVERSIONS__TORCH_CONVERSIONS_HPP_
