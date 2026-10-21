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

#ifndef ONNXRUNTIME_CONVERSIONS__CONVERSION_PLUGIN_HPP_
#define ONNXRUNTIME_CONVERSIONS__CONVERSION_PLUGIN_HPP_

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <memory>
#include <string>

#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace onnxruntime_conversions
{

struct ConversionView
{
  std::shared_ptr<void> lease;
  Ort::Value value{nullptr};
};

/// ONNX Runtime-specific conversion ABI implemented by packaged plugins.
class ConversionPlugin
{
public:
  using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

  virtual ~ConversionPlugin() = default;
  virtual std::string backend() const = 0;
  virtual bool supports(const Ort::ConstMemoryInfo & memory) const = 0;
  virtual bool available() const = 0;
  virtual int priority() const = 0;
  virtual void allocate(TensorMsg & msg, size_t byte_count, int device_id) = 0;
  virtual ConversionView from_input(
    const TensorMsg & msg, void * execution_stream) = 0;
  virtual ConversionView from_output(
    TensorMsg & msg, void * execution_stream) = 0;
  virtual void copy_to(
    TensorMsg & msg, const Ort::Value & source, size_t byte_count,
    void * execution_stream) = 0;
  virtual void configure_session(
    Ort::SessionOptions & options, int device_id,
    void * execution_stream) = 0;
  virtual Ort::SyncStream create_stream(Ort::Env & env, int device_id) = 0;
  virtual void validate_stream(int device_id, void * execution_stream) const = 0;
};

}  // namespace onnxruntime_conversions

#endif  // ONNXRUNTIME_CONVERSIONS__CONVERSION_PLUGIN_HPP_
