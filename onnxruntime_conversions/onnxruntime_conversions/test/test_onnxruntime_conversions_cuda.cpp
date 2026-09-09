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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

namespace
{

using onnxruntime_conversions::TensorMsg;
using onnxruntime_conversions::allocate_tensor_msg;
using onnxruntime_conversions::available_backends;
using onnxruntime_conversions::configure_session_options;
using onnxruntime_conversions::from_input_tensor_msg;
using onnxruntime_conversions::from_output_tensor_msg;
using onnxruntime_conversions::to_tensor_msg;

const uint8_t identity_model[] = {
  8, 10, 58, 88, 10, 25, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111,
  117, 116, 112, 117, 116, 34, 8, 73, 100, 101, 110, 116, 105, 116, 121,
  18, 8, 105, 100, 101, 110, 116, 105, 116, 121, 90, 23, 10, 5, 105,
  110, 112, 117, 116, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 2, 10,
  2, 8, 3, 98, 24, 10, 6, 111, 117, 116, 112, 117, 116, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 3, 66, 4, 10, 0, 16, 18};

class CudaConversions : public ::testing::Test
{
protected:
  void SetUp() override
  {
    const auto backends = available_backends();
    if (std::find(backends.begin(), backends.end(), "cuda") == backends.end()) {
      GTEST_SKIP() << "the CUDA storage plugin is unavailable";
    }
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override
  {
    if (stream_ != nullptr) {
      EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
    }
  }

  void * stream() const {return static_cast<void *>(stream_);}

  cudaStream_t stream_{nullptr};
};

TEST_F(CudaConversions, CudaIsPreferredOverHostMemory)
{
  EXPECT_EQ(onnxruntime_conversions::default_backend(), "cuda");

  auto msg = allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  EXPECT_EQ(msg->data.get_backend_type(), "cuda");
}

TEST_F(CudaConversions, HostStorageStaysAvailableAlongsideCuda)
{
  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");

  EXPECT_EQ(msg->data.get_backend_type(), "cpu");
}

TEST_F(CudaConversions, ViewsCarryTheCudaAllocatorAndDevicePointer)
{
  auto msg = allocate_tensor_msg(
    {6}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");

  auto output = from_output_tensor_msg(*msg, stream());
  const auto info = output.value().GetTensorMemoryInfo();

  EXPECT_EQ(info.GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  EXPECT_EQ(info.GetAllocatorName(), "Cuda");

  cudaPointerAttributes attributes{};
  ASSERT_EQ(
    cudaPointerGetAttributes(
      &attributes, output.value().GetTensorMutableData<float>()),
    cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
  EXPECT_EQ(attributes.device, info.GetDeviceId());
}

TEST_F(CudaConversions, InputAndOutputViewsAliasTheSameDeviceStorage)
{
  auto msg = allocate_tensor_msg(
    {6}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");

  auto output = from_output_tensor_msg(*msg, stream());
  auto * device_pointer = output.value().GetTensorMutableData<float>();
  auto input = from_input_tensor_msg(*msg, stream());

  EXPECT_EQ(input.value().GetTensorData<float>(), device_pointer);
}

TEST_F(CudaConversions, RoundTripsThroughDeviceMemory)
{
  const std::vector<float> source{1.0F, 2.0F, 3.0F, 4.0F};
  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");

  auto output = from_output_tensor_msg(*msg, stream());
  ASSERT_EQ(
    cudaMemcpyAsync(
      output.value().GetTensorMutableData<float>(), source.data(),
      source.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::vector<float> result(source.size());
  auto input = from_input_tensor_msg(*msg, stream());
  ASSERT_EQ(
    cudaMemcpyAsync(
      result.data(), input.value().GetTensorData<float>(),
      result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  EXPECT_EQ(result, source);
}

TEST_F(CudaConversions, LeaseOutlivesTheAcquiringScope)
{
  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  auto view = from_output_tensor_msg(*msg, stream());

  // The view holds the storage lease, so the mapping stays valid even after
  // another lease on the same message has been taken and dropped.
  { from_input_tensor_msg(*msg, stream()); }

  cudaPointerAttributes attributes{};
  EXPECT_EQ(
    cudaPointerGetAttributes(
      &attributes, view.value().GetTensorMutableData<float>()),
    cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
}

TEST_F(CudaConversions, CopiesDeviceOrtValueWithoutHostStaging)
{
  auto source = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  const std::vector<float> host{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  auto source_view = from_output_tensor_msg(*source, stream());
  ASSERT_EQ(
    cudaMemcpyAsync(
      source_view.value().GetTensorMutableData<float>(), host.data(),
      host.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  auto msg = to_tensor_msg(source_view.value(), stream());
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  EXPECT_EQ(msg->data.get_backend_type(), "cuda");
  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 3}));
  std::vector<float> result(host.size());
  auto result_view = from_input_tensor_msg(*msg, stream());
  ASSERT_EQ(
    cudaMemcpyAsync(
      result.data(), result_view.value().GetTensorData<float>(),
      result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  EXPECT_EQ(result, host);
}

TEST_F(CudaConversions, CopiesDeviceOrtValueIntoHostStorage)
{
  auto source = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  const std::vector<float> host{9.0F, 8.0F, 7.0F, 6.0F};
  auto source_view = from_output_tensor_msg(*source, stream());
  ASSERT_EQ(
    cudaMemcpyAsync(
      source_view.value().GetTensorMutableData<float>(), host.data(),
      host.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  to_tensor_msg(*msg, source_view.value(), stream());
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  const auto * result = reinterpret_cast<const float *>(msg->data.data());
  EXPECT_EQ(result[0], 9.0F);
  EXPECT_EQ(result[3], 6.0F);
}

/// The storage plugin and the execution provider are independent: device
/// memory can be shared even when the linked ONNX Runtime has no CUDA
/// provider, so inference is the only part that needs one.
TEST_F(CudaConversions, ConfiguresTheProviderAndRunsInferenceOnDeviceStorage)
{
  const auto providers = Ort::GetAvailableProviders();
  if (std::find(
      providers.begin(), providers.end(),
      "CUDAExecutionProvider") == providers.end())
  {
    GTEST_SKIP() << "this ONNX Runtime build has no CUDA execution provider";
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_conversions_cuda_test");
  Ort::SessionOptions session_options;
  configure_session_options(session_options, "cuda", 0, stream());
  Ort::Session session(
    env, identity_model, sizeof(identity_model), session_options);
  Ort::IoBinding binding(session);

  const std::vector<float> host{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  auto input = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  auto output = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");

  auto input_view = from_input_tensor_msg(*input, stream());
  auto output_view = from_output_tensor_msg(*output, stream());
  ASSERT_EQ(
    cudaMemcpyAsync(
      const_cast<float *>(input_view.value().GetTensorData<float>()),
      host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice,
      stream_),
    cudaSuccess);

  binding.BindInput("input", input_view.value());
  binding.BindOutput("output", output_view.value());
  session.Run(Ort::RunOptions{}, binding);
  binding.SynchronizeOutputs();

  std::vector<float> result(host.size());
  ASSERT_EQ(
    cudaMemcpyAsync(
      result.data(), output_view.value().GetTensorData<float>(),
      result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  EXPECT_EQ(result, host);
}

}  // namespace
