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

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

#ifdef ONNXRUNTIME_CONVERSIONS_TEST_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace
{

using onnxruntime_conversions::TensorMsg;
using onnxruntime_conversions::allocate_tensor_msg;
using onnxruntime_conversions::from_input_tensor_msg;
using onnxruntime_conversions::from_output_tensor_msg;
using onnxruntime_conversions::to_tensor_msg;

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

TEST(OnnxRuntimeConversions, AllocatePopulatesMetadata)
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

TEST(OnnxRuntimeConversions, OutputViewAliasesMessageStorage)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto memory_info = cpu_memory_info();
  auto view = from_output_tensor_msg(msg, memory_info);

  auto * data = view.value().GetTensorMutableData<int32_t>();
  EXPECT_EQ(
    static_cast<void *>(data),
    static_cast<void *>(msg->data.data()));

  data[0] = 10;
  data[1] = 20;
  data[2] = 30;
  data[3] = 40;
  const auto * message_data =
    reinterpret_cast<const int32_t *>(msg->data.data());
  EXPECT_EQ(message_data[0], 10);
  EXPECT_EQ(message_data[3], 40);
}

TEST(OnnxRuntimeConversions, SupportsScalarAndZeroSizedShapes)
{
  auto memory_info = cpu_memory_info();
  std::shared_ptr<TensorMsg> scalar = allocate_tensor_msg(
    {}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  auto scalar_view = from_output_tensor_msg(scalar, memory_info);
  EXPECT_EQ(
    scalar_view.value().GetTensorTypeAndShapeInfo().GetElementCount(), 1u);

  std::shared_ptr<TensorMsg> empty = allocate_tensor_msg(
    {2, 0, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  auto empty_view = from_output_tensor_msg(empty, memory_info);
  EXPECT_EQ(
    empty_view.value().GetTensorTypeAndShapeInfo().GetElementCount(), 0u);
  EXPECT_EQ(
    empty_view.value().GetTensorTypeAndShapeInfo().GetShape(),
    (std::vector<int64_t>{2, 0, 3}));
}

TEST(OnnxRuntimeConversions, InputViewAliasesMessageStorageAtByteOffset)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {8}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto * data = reinterpret_cast<int32_t *>(msg->data.data());
  for (int32_t index = 0; index < 8; ++index) {
    data[index] = index * 10;
  }
  msg->shape = {3};
  msg->strides = {1};
  msg->byte_offset = 2 * sizeof(int32_t);

  auto memory_info = cpu_memory_info();
  std::shared_ptr<const TensorMsg> input = msg;
  auto view = from_input_tensor_msg(input, memory_info);
  const auto * view_data = view.value().GetTensorData<int32_t>();

  EXPECT_EQ(view_data, data + 2);
  EXPECT_EQ(view_data[0], 20);
  EXPECT_EQ(view_data[2], 40);
}

TEST(OnnxRuntimeConversions, ViewKeepsMessageAlive)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::weak_ptr<TensorMsg> weak = msg;
  auto memory_info = cpu_memory_info();

  {
    auto view = from_output_tensor_msg(msg, memory_info);
    msg.reset();
    EXPECT_FALSE(weak.expired());
    view.value().GetTensorMutableData<float>()[0] = 3.0F;
  }

  EXPECT_TRUE(weak.expired());
}

TEST(OnnxRuntimeConversions, RejectsNonContiguousStrides)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  msg->strides = {1, 2};
  auto memory_info = cpu_memory_info();

  EXPECT_THROW(
    from_output_tensor_msg(msg, memory_info),
    std::invalid_argument);
}

TEST(OnnxRuntimeConversions, RejectsOutOfBoundsView)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  msg->byte_offset = sizeof(float);
  auto memory_info = cpu_memory_info();

  EXPECT_THROW(
    from_output_tensor_msg(msg, memory_info),
    std::out_of_range);
}

TEST(OnnxRuntimeConversions, RejectsMismatchedMemoryInfo)
{
  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  Ort::MemoryInfo cuda_memory_info(
    "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  EXPECT_THROW(
    from_output_tensor_msg(msg, cuda_memory_info),
    std::invalid_argument);
}

TEST(OnnxRuntimeConversions, CopiesCpuOrtValueIntoMessage)
{
  std::vector<float> source{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<int64_t> shape{2, 3};
  auto memory_info = cpu_memory_info();
  auto value = Ort::Value::CreateTensor<float>(
    memory_info, source.data(), source.size(), shape.data(), shape.size());

  auto msg = to_tensor_msg(value);
  EXPECT_EQ(msg->shape, shape);
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{3, 1}));
  EXPECT_EQ(msg->dtype_code, 2);
  EXPECT_EQ(msg->dtype_bits, 32);

  const auto * result =
    reinterpret_cast<const float *>(msg->data.data());
  EXPECT_EQ(result[0], 1.0F);
  EXPECT_EQ(result[5], 6.0F);
}

TEST(OnnxRuntimeConversions, RunsInferenceWithPreallocatedMessageBuffers)
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_conversions_test");
  Ort::SessionOptions session_options;
  Ort::Session session(
    env, identity_model, sizeof(identity_model), session_options);
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

  const auto * output_data =
    reinterpret_cast<const float *>(output->data.data());
  for (size_t index = 0; index < 6; ++index) {
    EXPECT_EQ(output_data[index], input_data[index]);
  }
  EXPECT_EQ(
    output_view.value().GetTensorMutableData<float>(),
    reinterpret_cast<float *>(output->data.data()));
}

#ifdef ONNXRUNTIME_CONVERSIONS_TEST_HAS_CUDA
TEST(OnnxRuntimeConversions, CudaViewsAliasStorageAndSynchronize)
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
    ASSERT_NE(device_pointer, nullptr);
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

#ifdef ONNXRUNTIME_CONVERSIONS_TEST_HAS_CUDA_EP
TEST(OnnxRuntimeConversions, RunsCudaInferenceWithMessageBuffers)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_conversions_cuda_test");
  Ort::CUDAProviderOptions cuda_options;
  cuda_options.UpdateWithValue("user_compute_stream", stream);
  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  Ort::Session session(
    env, identity_model, sizeof(identity_model), session_options);
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
#endif
#endif

TEST(OnnxRuntimeConversions, RejectsUnsupportedDtypeAndInvalidMetadata)
{
  EXPECT_THROW(
    allocate_tensor_msg({2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
    std::invalid_argument);

  std::shared_ptr<TensorMsg> msg = allocate_tensor_msg(
    {2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  msg->dtype_lanes = 2;
  auto memory_info = cpu_memory_info();
  EXPECT_THROW(
    from_output_tensor_msg(msg, memory_info),
    std::invalid_argument);
}

}  // namespace
