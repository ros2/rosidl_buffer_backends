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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "onnxruntime_cuda_conversions/onnxruntime_cuda_conversions.hpp"

namespace
{

using onnxruntime_cuda_conversions::TensorMsg;
using onnxruntime_cuda_conversions::allocate_tensor_msg;
using onnxruntime_cuda_conversions::from_input_tensor_msg;
using onnxruntime_cuda_conversions::from_output_tensor_msg;
using onnxruntime_cuda_conversions::to_tensor_msg;

Ort::MemoryInfo cpu_memory_info()
{
  return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

const uint8_t identity_model[] = {
  8, 10, 58, 88, 10, 25, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111,
  117, 116, 112, 117, 116, 34, 8, 73, 100, 101, 110, 116, 105, 116, 121,
  18, 8, 105, 100, 101, 110, 116, 105, 116, 121, 90, 23, 10, 5, 105,
  110, 112, 117, 116, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 2, 10,
  2, 8, 3, 98, 24, 10, 6, 111, 117, 116, 112, 117, 116, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 3, 66, 4, 10, 0, 16, 18};

TEST(OnnxRuntimeCudaConversions, AllocatePopulatesCpuMetadata)
{
  auto msg = allocate_tensor_msg(
    {2, 3, 4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{12, 4, 1}));
  EXPECT_EQ(msg->dtype_code, 2);
  EXPECT_EQ(msg->dtype_bits, 32);
  EXPECT_EQ(msg->dtype_lanes, 1);
  EXPECT_EQ(msg->byte_offset, 0u);
  EXPECT_EQ(msg->data.size(), 24u * sizeof(float));
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");
}

TEST(OnnxRuntimeCudaConversions, CpuViewsAliasMessageStorage)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto memory_info = cpu_memory_info();
  auto output = from_output_tensor_msg(msg, memory_info);
  auto * data = output.value().GetTensorMutableData<int32_t>();
  EXPECT_EQ(static_cast<void *>(data), static_cast<void *>(msg->data.data()));

  data[0] = 10;
  data[3] = 40;
  std::shared_ptr<const TensorMsg> input_msg = msg;
  auto input = from_input_tensor_msg(input_msg, memory_info);
  EXPECT_EQ(input.value().GetTensorData<int32_t>(), data);
  EXPECT_EQ(input.value().GetTensorData<int32_t>()[3], 40);
}

TEST(OnnxRuntimeCudaConversions, SupportsScalarZeroSizedAndOffsetViews)
{
  auto memory_info = cpu_memory_info();
  std::shared_ptr<TensorMsg> scalar = allocate_tensor_msg(
    {}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  auto scalar_view = from_output_tensor_msg(scalar, memory_info);
  EXPECT_EQ(scalar_view.value().GetTensorTypeAndShapeInfo().GetElementCount(), 1u);

  std::shared_ptr<TensorMsg> empty = allocate_tensor_msg(
    {2, 0, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  auto empty_view = from_output_tensor_msg(empty, memory_info);
  EXPECT_EQ(empty_view.value().GetTensorTypeAndShapeInfo().GetElementCount(), 0u);

  std::shared_ptr<TensorMsg> offset = allocate_tensor_msg(
    {8}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto * offset_data = reinterpret_cast<int32_t *>(offset->data.data());
  offset->shape = {3};
  offset->strides = {1};
  offset->byte_offset = 2 * sizeof(int32_t);
  std::shared_ptr<const TensorMsg> offset_input = offset;
  auto offset_view = from_input_tensor_msg(offset_input, memory_info);
  EXPECT_EQ(offset_view.value().GetTensorData<int32_t>(), offset_data + 2);
}

TEST(OnnxRuntimeCudaConversions, ViewKeepsMessageAlive)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::weak_ptr<TensorMsg> weak = msg;
  auto memory_info = cpu_memory_info();
  {
    auto view = from_output_tensor_msg(msg, memory_info);
    msg.reset();
    EXPECT_FALSE(weak.expired());
  }
  EXPECT_TRUE(weak.expired());
}

TEST(OnnxRuntimeCudaConversions, RejectsInvalidCpuMetadata)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  msg->strides = {1, 2};
  auto memory_info = cpu_memory_info();
  EXPECT_THROW(from_output_tensor_msg(msg, memory_info), std::invalid_argument);

  msg = allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  msg->byte_offset = sizeof(float);
  EXPECT_THROW(from_output_tensor_msg(msg, memory_info), std::out_of_range);

  Ort::MemoryInfo cuda_memory_info(
    "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  msg->byte_offset = 0;
  EXPECT_THROW(from_output_tensor_msg(msg, cuda_memory_info), std::invalid_argument);
}

TEST(OnnxRuntimeCudaConversions, CopiesCpuOrtValueIntoMessage)
{
  std::vector<float> source{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<int64_t> shape{2, 3};
  auto memory_info = cpu_memory_info();
  auto value = Ort::Value::CreateTensor<float>(
    memory_info, source.data(), source.size(), shape.data(), shape.size());

  auto msg = to_tensor_msg(value);
  EXPECT_EQ(msg->shape, shape);
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{3, 1}));
  const auto * result = reinterpret_cast<const float *>(msg->data.data());
  EXPECT_EQ(result[0], 1.0F);
  EXPECT_EQ(result[5], 6.0F);
}

TEST(OnnxRuntimeCudaConversions, RunsCpuInferenceWithMessageBuffers)
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_cuda_conversions_cpu_test");
  Ort::SessionOptions session_options;
  Ort::Session session(env, identity_model, sizeof(identity_model), session_options);
  Ort::IoBinding binding(session);
  auto memory_info = cpu_memory_info();
  std::shared_ptr<TensorMsg> input = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::shared_ptr<TensorMsg> output = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  auto * input_data = reinterpret_cast<float *>(input->data.data());
  for (size_t index = 0; index < 6; ++index) {
    input_data[index] = static_cast<float>(index + 1);
  }

  std::shared_ptr<const TensorMsg> const_input = input;
  auto input_view = from_input_tensor_msg(const_input, memory_info);
  auto output_view = from_output_tensor_msg(output, memory_info);
  binding.BindInput("input", input_view.value());
  binding.BindOutput("output", output_view.value());
  Ort::RunOptions run_options;
  session.Run(run_options, binding);

  const auto * output_data = reinterpret_cast<const float *>(output->data.data());
  for (size_t index = 0; index < 6; ++index) {
    EXPECT_EQ(output_data[index], input_data[index]);
  }
}

TEST(OnnxRuntimeCudaConversions, CudaViewsAliasStorageAndSynchronize)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  const std::vector<float> expected{1.0F, 2.0F, 3.0F, 4.0F};
  void * device_pointer = nullptr;
  {
    auto output = from_output_tensor_msg(msg, memory_info, stream);
    device_pointer = output.value().GetTensorMutableRawData();
    ASSERT_EQ(
      cudaMemcpyAsync(
        device_pointer, expected.data(), expected.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream),
      cudaSuccess);
  }

  std::vector<float> actual(expected.size());
  {
    std::shared_ptr<const TensorMsg> input_msg = msg;
    auto input = from_input_tensor_msg(input_msg, memory_info, stream);
    EXPECT_EQ(input.value().GetTensorRawData(), device_pointer);
    ASSERT_EQ(
      cudaMemcpyAsync(
        actual.data(), input.value().GetTensorRawData(),
        actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
      cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(OnnxRuntimeCudaConversions, RunsCudaInferenceWithMessageBuffers)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_cuda_conversions_cuda_test");
  Ort::CUDAProviderOptions cuda_options;
  cuda_options.UpdateWithValue("user_compute_stream", stream);
  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  Ort::Session session(env, identity_model, sizeof(identity_model), session_options);
  Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::shared_ptr<TensorMsg> input = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  std::shared_ptr<TensorMsg> output = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  const std::vector<float> expected{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  {
    auto writer = from_output_tensor_msg(input, memory_info, stream);
    ASSERT_EQ(
      cudaMemcpyAsync(
        writer.value().GetTensorMutableRawData(), expected.data(),
        expected.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
      cudaSuccess);
  }
  {
    Ort::IoBinding binding(session);
    std::shared_ptr<const TensorMsg> const_input = input;
    auto input_view = from_input_tensor_msg(const_input, memory_info, stream);
    auto output_view = from_output_tensor_msg(output, memory_info, stream);
    binding.BindInput("input", input_view.value());
    binding.BindOutput("output", output_view.value());
    Ort::RunOptions run_options;
    session.Run(run_options, binding);
  }

  std::vector<float> actual(expected.size());
  {
    std::shared_ptr<const TensorMsg> const_output = output;
    auto reader = from_input_tensor_msg(const_output, memory_info, stream);
    ASSERT_EQ(
      cudaMemcpyAsync(
        actual.data(), reader.value().GetTensorRawData(),
        actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
      cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace
