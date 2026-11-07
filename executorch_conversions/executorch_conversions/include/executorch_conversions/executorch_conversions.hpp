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

#ifndef EXECUTORCH_CONVERSIONS__EXECUTORCH_CONVERSIONS_HPP_
#define EXECUTORCH_CONVERSIONS__EXECUTORCH_CONVERSIONS_HPP_

#include <executorch/extension/tensor/tensor_ptr.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensor_msgs/msg/experimental_tensor.hpp"
#include "executorch_conversions/visibility_control.hpp"

namespace executorch_conversions
{

using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

EXECUTORCH_CONVERSIONS_PUBLIC std::vector<std::string> available_backends();

EXECUTORCH_CONVERSIONS_PUBLIC bool backend_available(
  const std::string & backend);

/// Throws when no available plugin serves device_type.
EXECUTORCH_CONVERSIONS_PUBLIC std::string backend_for_device(
  executorch::aten::DeviceType device_type);

/// Throws when no plugin is available.
EXECUTORCH_CONVERSIONS_PUBLIC std::string default_backend();

EXECUTORCH_CONVERSIONS_PUBLIC executorch::aten::ScalarType scalar_type(
  const TensorMsg & msg);

/// Returns msg's strides as given, or contiguous ones when it has none. Throws when their
/// count differs from the rank; their values are not checked.
EXECUTORCH_CONVERSIONS_PUBLIC std::vector<int64_t> normalized_strides(
  const TensorMsg & msg);

/// Allocates a contiguous message. Without a device, it uses the highest-priority
/// available plugin, which is CUDA when that plugin is installed and a GPU is usable.
/// A CUDA device without an index, and that default, mean GPU 0, not the current device.
/// cuda_buffer serves one GPU per process: the current one when the process creates its
/// first CUDA buffer, publisher or subscription.
EXECUTORCH_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  executorch::aten::ScalarType dtype,
  std::optional<executorch::aten::Device> device = std::nullopt);

/// Views borrow msg storage. msg must outlive the view, and the view must be released
/// before publishing. Returns null when msg has no data, and throws for a message it
/// cannot view, for example one that is not contiguous.
/// A CUDA message accepts one write: one output view or one to_tensor_msg. Do not view
/// it as output while its input view is alive: the write is not ordered after those reads.
/// After the view is released, the write throws until those reads finish on the GPU, so
/// synchronize the reading stream first.
/// A CUDA view is ordered on execution_stream, which must outlive it, or else on the
/// stream ExecuTorch's CUDA backend would pick at this call (see the README's Streams).
/// Throws when that stream is on another GPU than msg.
EXECUTORCH_CONVERSIONS_PUBLIC executorch::extension::TensorPtr from_output_tensor_msg(
  TensorMsg & msg,
  void * execution_stream = nullptr);

/// Views borrow msg storage and must be treated as read-only.
/// msg must outlive the view and keep its size while the view is used.
/// Do not view a CUDA message as input while its output view is alive: the first read
/// ends the write, so later writes through the output view are not ordered.
/// Throws for a message it cannot view, for example one that is not contiguous.
/// Returns null when msg has no data. Streams follow from_output_tensor_msg.
EXECUTORCH_CONVERSIONS_PUBLIC executorch::extension::TensorPtr from_input_tensor_msg(
  const TensorMsg & msg,
  void * execution_stream = nullptr);

/// Copies tensor into msg storage, which must be large enough. Throws for a tensor
/// that is not contiguous. An empty tensor leaves msg unchanged. A CUDA message
/// accepts one write, and throws when tensor overlaps the bytes it would write.
EXECUTORCH_CONVERSIONS_PUBLIC void to_tensor_msg(
  TensorMsg & msg,
  const executorch::aten::Tensor & tensor,
  void * execution_stream = nullptr);

/// Copies tensor into a new message on its device. Throws for a tensor that is not
/// contiguous. An empty tensor gives an empty message.
EXECUTORCH_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> to_tensor_msg(
  const executorch::aten::Tensor & tensor,
  void * execution_stream = nullptr);

}  // namespace executorch_conversions

#endif  // EXECUTORCH_CONVERSIONS__EXECUTORCH_CONVERSIONS_HPP_
