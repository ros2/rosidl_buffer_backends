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

#ifndef ONNXRUNTIME_CONVERSIONS__CONVERSION_BACKEND_HPP_
#define ONNXRUNTIME_CONVERSIONS__CONVERSION_BACKEND_HPP_

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_conversions/visibility_control.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace onnxruntime_conversions
{

using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

struct StorageMetadata
{
  void * data{nullptr};
  size_t size_bytes{0};
  OrtMemoryInfoDeviceType device_type{OrtMemoryInfoDeviceType_CPU};
  int device_id{0};
  std::string allocator_name;
};

class ONNXRUNTIME_CONVERSIONS_PUBLIC StorageLease
{
public:
  virtual ~StorageLease();
  virtual const StorageMetadata & metadata() const noexcept = 0;
};

struct BackendConfiguration
{
  int device_id{0};
  void * execution_stream{nullptr};
};

class ONNXRUNTIME_CONVERSIONS_PUBLIC ConversionBackend
{
public:
  virtual ~ConversionBackend();

  virtual std::string backend_name() const = 0;
  virtual void allocate_storage(TensorMsg & msg, size_t byte_count) = 0;
  virtual std::shared_ptr<StorageLease> acquire_input(
    std::shared_ptr<const TensorMsg> msg,
    void * execution_stream) = 0;
  virtual std::shared_ptr<StorageLease> acquire_output(
    std::shared_ptr<TensorMsg> msg,
    void * execution_stream) = 0;
  virtual void copy_from_ort(
    TensorMsg & msg,
    const Ort::Value & value,
    size_t byte_count,
    void * execution_stream) = 0;
  virtual void configure_session(
    Ort::SessionOptions & session_options,
    const BackendConfiguration & configuration) = 0;
};

class ONNXRUNTIME_CONVERSIONS_PUBLIC AutomaticSelectionCapability
{
public:
  virtual ~AutomaticSelectionCapability();
  virtual bool supports_automatic_selection(
    const BackendConfiguration & configuration) const noexcept = 0;
};

class ONNXRUNTIME_CONVERSIONS_PUBLIC ConversionBackendRegistry
{
public:
  static ConversionBackendRegistry & instance();

  std::shared_ptr<ConversionBackend> get_backend(const std::string & backend);
  std::shared_ptr<ConversionBackend> select_backend(
    const BackendConfiguration & configuration);
  std::vector<std::string> available_backends() const;

  ConversionBackendRegistry(const ConversionBackendRegistry &) = delete;
  ConversionBackendRegistry & operator=(const ConversionBackendRegistry &) = delete;

private:
  ConversionBackendRegistry();
  ~ConversionBackendRegistry();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace onnxruntime_conversions

#endif  // ONNXRUNTIME_CONVERSIONS__CONVERSION_BACKEND_HPP_
