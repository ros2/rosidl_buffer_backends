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
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

namespace
{

using onnxruntime_conversions::TensorMsg;
using onnxruntime_conversions::allocate_tensor_msg;
using onnxruntime_conversions::backend_available;
using onnxruntime_conversions::borrow_stream;
using onnxruntime_conversions::configure_session_options;
using onnxruntime_conversions::create_stream;
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

// MatMul(input, input), float32 [2, 2], IR 10, opset 18.
const uint8_t matmul_model[] = {
  8, 10, 58, 93, 10, 30, 10, 5, 105, 110, 112, 117, 116,
  10, 5, 105, 110, 112, 117, 116, 18, 6, 111,
  117, 116, 112, 117, 116, 34, 6, 77, 97, 116, 77, 117, 108,
  18, 8, 105, 100, 101, 110, 116, 105, 116, 121, 90, 23, 10, 5, 105,
  110, 112, 117, 116, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 2, 10,
  2, 8, 2, 98, 24, 10, 6, 111, 117, 116, 112, 117, 116, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 2, 66, 4, 10, 0, 16, 18};

class CudaConversions : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!backend_available("cuda")) {
      GTEST_SKIP() << "the CUDA conversion plugin is unavailable";
    }
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
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

void copy_to_device(
  Ort::Value & value, const std::vector<float> & source, cudaStream_t stream)
{
  ASSERT_EQ(
    cudaMemcpyAsync(
      value.GetTensorMutableRawData(), source.data(),
      source.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}

void expect_device_values(
  const Ort::Value & value, const std::vector<float> & expected,
  cudaStream_t stream)
{
  std::vector<float> actual(expected.size());
  ASSERT_EQ(
    cudaMemcpyAsync(
      actual.data(), value.GetTensorRawData(), actual.size() * sizeof(float),
      cudaMemcpyDeviceToHost, stream),
    cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(actual, expected);
}

TEST_F(CudaConversions, PrefersCudaAndKeepsHostStorageAvailable)
{
  EXPECT_EQ(onnxruntime_conversions::default_backend(), "cuda");

  auto default_msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  auto host_msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");

  EXPECT_EQ(default_msg->data.get_backend_type(), "cuda");
  EXPECT_EQ(host_msg->data.get_backend_type(), "cpu");
}

TEST_F(CudaConversions, StreamOwnershipAndDeviceValidation)
{
  for (int iteration = 0; iteration < 2; ++iteration) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "stream_ownership_test");
    auto retained = [&env]() {
        auto owner = create_stream(env);
        EXPECT_TRUE(owner.owns_stream());
        auto copy = owner;
        return copy;
      }();
    EXPECT_EQ(retained.backend(), "cuda");
    EXPECT_EQ(retained.device_id(), 0);
    auto second = create_stream(env, "cuda");
    EXPECT_NE(retained.handle(), second.handle());
    EXPECT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(retained.handle())), cudaSuccess);
    {
      auto borrowed = borrow_stream(stream(), "cuda");
      EXPECT_FALSE(borrowed.owns_stream());
      EXPECT_EQ(borrowed.handle(), stream());
    }
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_THROW(borrow_stream(nullptr, "cuda"), std::invalid_argument);
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    EXPECT_THROW(create_stream(env, "cuda", count), std::runtime_error);
    EXPECT_THROW(borrow_stream(stream(), "cuda", count), std::runtime_error);
  }
}

TEST_F(CudaConversions, OwnedAndBorrowedStreamsOrderMatMulInference)
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "stream_matmul_test");
  for (bool own : {true, false}) {
    auto selected = own ? create_stream(env) : borrow_stream(stream(), "cuda");
    const auto native = static_cast<cudaStream_t>(selected.handle());
    Ort::SessionOptions options;
    configure_session_options(options, selected);
    Ort::Session session(env, matmul_model, sizeof(matmul_model), options);
    auto input = allocate_tensor_msg({2, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, selected);
    auto output = allocate_tensor_msg({2, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, selected);
    const std::vector<float> values(4, 7.0F);
    {
      auto view = from_output_tensor_msg(*input, selected);
      ASSERT_EQ(cudaMemsetAsync(view.value().GetTensorMutableRawData(), 0, 16, native),
        cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(native), cudaSuccess);
      ASSERT_EQ(cudaLaunchHostFunc(native, [](void *) {
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }, nullptr), cudaSuccess);
      ASSERT_EQ(cudaMemcpyAsync(view.value().GetTensorMutableRawData(), values.data(), 16,
        cudaMemcpyHostToDevice, native), cudaSuccess);
    }
    auto input_view = from_input_tensor_msg(*input, selected);
    auto output_view = from_output_tensor_msg(*output, selected);
    Ort::IoBinding binding(session);
    binding.BindInput("input", input_view.value());
    binding.BindOutput("output", output_view.value());
    session.Run(Ort::RunOptions{}, binding);
    expect_device_values(output_view.value(), {98, 98, 98, 98}, native);
  }
}

TEST_F(CudaConversions, ViewsAliasCudaDeviceStorage)
{
  const std::vector<float> source{1.0F, 2.0F, 3.0F, 4.0F};
  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");

  auto output = from_output_tensor_msg(*msg, stream());
  const auto info = output.value().GetTensorMemoryInfo();
  EXPECT_EQ(info.GetDeviceType(), OrtMemoryInfoDeviceType_GPU);
  EXPECT_EQ(info.GetAllocatorName(), "Cuda");

  auto * device_pointer = output.value().GetTensorMutableData<float>();
  cudaPointerAttributes attributes{};
  ASSERT_EQ(
    cudaPointerGetAttributes(&attributes, device_pointer),
    cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
  EXPECT_EQ(attributes.device, info.GetDeviceId());

  copy_to_device(output.value(), source, stream_);
  auto input = from_input_tensor_msg(*msg, stream());
  EXPECT_EQ(input.value().GetTensorData<float>(), device_pointer);
  expect_device_values(input.value(), source, stream_);
}

TEST_F(CudaConversions, ViewsWaitForTheProducerOnTheConsumerStream)
{
  cudaStream_t producer = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
  auto msg = allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "cuda");
  {
    auto output = from_output_tensor_msg(*msg, producer);
    ASSERT_EQ(cudaLaunchHostFunc(producer, [](void *) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }, nullptr), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(output.value().GetTensorMutableRawData(), 1, 16, producer),
      cudaSuccess);
  }
  {
    auto input = from_input_tensor_msg(*msg, stream());
    int32_t actual[4]{};
    ASSERT_EQ(cudaMemcpyAsync(actual, input.value().GetTensorRawData(), sizeof(actual),
      cudaMemcpyDeviceToHost, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    for (const auto element : actual) {
      EXPECT_EQ(element, 0x01010101);
    }
  }
  ASSERT_EQ(cudaStreamDestroy(producer), cudaSuccess);
}

TEST_F(CudaConversions, CopiesCompleteBeforeSourceStorageIsReleased)
{
  auto source = allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  auto destination = allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  {
    auto view = from_output_tensor_msg(*source, stream());
    copy_to_device(view.value(), {1, 2, 3, 4}, stream_);
    ASSERT_EQ(cudaLaunchHostFunc(stream_, [](void *) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }, nullptr), cudaSuccess);
    to_tensor_msg(*destination, view.value(), stream());
    EXPECT_EQ(cudaStreamQuery(stream_), cudaSuccess);
  }
  source.reset();
  auto result = from_input_tensor_msg(*destination, stream());
  expect_device_values(result.value(), {1, 2, 3, 4}, stream_);
}

TEST_F(CudaConversions, PreservesAndValidatesDeviceIndices)
{
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  EXPECT_THROW(allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda", count),
    std::runtime_error);
  int current = 0;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  for (int device = 0; device < count; ++device) {
    if (device != current) {
      EXPECT_THROW(allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda", device),
        std::runtime_error);
      continue;
    }
    auto msg = allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda", device);
    auto view = from_input_tensor_msg(*msg);
    EXPECT_EQ(view.value().GetTensorMemoryInfo().GetDeviceId(), device);
    auto copy = to_tensor_msg(view.value());
    auto copied_view = from_input_tensor_msg(*copy);
    EXPECT_EQ(copied_view.value().GetTensorMemoryInfo().GetDeviceId(), device);
  }
}

TEST_F(CudaConversions, CopiesDeviceOrtValueWithoutHostStaging)
{
  auto source = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  const std::vector<float> host{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  auto source_view = from_output_tensor_msg(*source, stream());
  copy_to_device(source_view.value(), host, stream_);

  auto msg = to_tensor_msg(source_view.value(), stream());
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  EXPECT_EQ(msg->data.get_backend_type(), "cuda");
  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 3}));
  auto result_view = from_input_tensor_msg(*msg, stream());
  expect_device_values(result_view.value(), host, stream_);
}

TEST_F(CudaConversions, CopiesDeviceOrtValueIntoHostStorage)
{
  auto source = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  const std::vector<float> host{9.0F, 8.0F, 7.0F, 6.0F};
  auto source_view = from_output_tensor_msg(*source, stream());
  copy_to_device(source_view.value(), host, stream_);

  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  to_tensor_msg(*msg, source_view.value(), stream());
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  const auto * result = reinterpret_cast<const float *>(msg->data.data());
  EXPECT_EQ(result[0], 9.0F);
  EXPECT_EQ(result[3], 6.0F);
}

TEST_F(CudaConversions, ConfiguresTheProviderAndRunsInferenceOnDeviceStorage)
{
  const auto providers = Ort::GetAvailableProviders();
  ASSERT_NE(
    std::find(providers.begin(), providers.end(), "CUDAExecutionProvider"),
    providers.end());

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
  copy_to_device(input_view.value(), host, stream_);

  binding.BindInput("input", input_view.value());
  binding.BindOutput("output", output_view.value());
  session.Run(Ort::RunOptions{}, binding);
  binding.SynchronizeOutputs();

  expect_device_values(output_view.value(), host, stream_);
}

}  // namespace
