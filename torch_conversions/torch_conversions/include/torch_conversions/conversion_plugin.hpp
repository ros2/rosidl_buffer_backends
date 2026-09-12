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

#ifndef TORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_
#define TORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_

#include <c10/core/Stream.h>
#include <torch/torch.h>

#include <cstddef>
#include <optional>
#include <string>

#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace torch_conversions
{

/// Torch-specific conversion ABI implemented by independently packaged DSOs.
class ConversionPlugin
{
public:
  using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

  virtual ~ConversionPlugin() = default;

  virtual std::string backend() const = 0;
  virtual c10::DeviceType device_type() const = 0;
  virtual bool available() const = 0;
  virtual int priority() const = 0;

  virtual std::optional<c10::Stream> select_stream(c10::Device device) = 0;

  virtual void allocate(
    TensorMsg & msg, size_t byte_count, c10::Device device) = 0;

  virtual at::Tensor from_input(
    const TensorMsg & msg, bool clone, void * execution_stream) = 0;

  virtual at::Tensor from_output(
    TensorMsg & msg, void * execution_stream) = 0;

  /// Copy into message storage; the source may be non-contiguous.
  virtual void copy_to(
    TensorMsg & msg,
    const at::Tensor & source,
    void * execution_stream) = 0;
};

}  // namespace torch_conversions

#endif  // TORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_
