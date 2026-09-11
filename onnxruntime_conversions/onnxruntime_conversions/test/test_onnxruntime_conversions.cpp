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

#include <algorithm>
#include <cstdint>
#include <limits>
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

bool installed(const std::string & backend)
{
  const auto backends = available_backends();
  return std::find(backends.begin(), backends.end(), backend) != backends.end();
}

Ort::Value host_value(
  std::vector<float> & data, const std::vector<int64_t> & shape)
{
  auto info = Ort::MemoryInfo::CreateCpu(
    OrtDeviceAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor<float>(
    info, data.data(), data.size(), shape.data(), shape.size());
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
  EXPECT_TRUE(installed("cpu"));
  auto msg = allocate_tensor_msg(
    {2, 3, 4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");

  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{12, 4, 1}));
  EXPECT_EQ(msg->dtype_code, 2);
  EXPECT_EQ(msg->dtype_bits, 32);
  EXPECT_EQ(msg->dtype_lanes, 1);
  EXPECT_EQ(msg->byte_offset, 0u);
  EXPECT_EQ(msg->data.size(), 24u * sizeof(float));
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");

  auto default_msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  EXPECT_EQ(
    default_msg->data.get_backend_type(),
    onnxruntime_conversions::default_backend());
}

TEST(OnnxRuntimeConversions, UnavailableBackendThrows)
{
  if (installed("cuda")) {
    GTEST_SKIP() << "the CUDA conversion plugin is installed";
  }
  EXPECT_THROW(
    allocate_tensor_msg({1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"),
    std::runtime_error);
}

TEST(OnnxRuntimeConversions, OutputViewAliasesMessageStorage)
{
  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "cpu");
  auto view = from_output_tensor_msg(*msg);

  auto * data = view.value().GetTensorMutableData<int32_t>();
  EXPECT_EQ(static_cast<void *>(data), static_cast<void *>(msg->data.data()));

  data[0] = 10;
  data[3] = 40;
  const auto * message_data =
    reinterpret_cast<const int32_t *>(msg->data.data());
  EXPECT_EQ(message_data[0], 10);
  EXPECT_EQ(message_data[3], 40);

  const auto info = view.value().GetTensorMemoryInfo();
  EXPECT_EQ(info.GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
  EXPECT_EQ(info.GetDeviceId(), 0);
}

TEST(OnnxRuntimeConversions, InputViewAliasesMessageStorageAtByteOffset)
{
  auto msg = allocate_tensor_msg(
    {8}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "cpu");
  auto * data = reinterpret_cast<int32_t *>(msg->data.data());
  for (int32_t index = 0; index < 8; ++index) {
    data[index] = index * 10;
  }
  msg->shape = {3};
  msg->strides = {1};
  msg->byte_offset = 2 * sizeof(int32_t);

  auto view = from_input_tensor_msg(*msg);
  const auto * view_data = view.value().GetTensorData<int32_t>();

  EXPECT_EQ(view_data, data + 2);
  EXPECT_EQ(view_data[0], 20);
  EXPECT_EQ(view_data[2], 40);
  EXPECT_EQ(
    view.value().GetTensorTypeAndShapeInfo().GetShape(),
    (std::vector<int64_t>{3}));
}

TEST(OnnxRuntimeConversions, SupportsScalarShapes)
{
  auto scalar = allocate_tensor_msg(
    {}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  auto view = from_output_tensor_msg(*scalar);

  EXPECT_TRUE(static_cast<bool>(view));
  EXPECT_EQ(view.value().GetTensorTypeAndShapeInfo().GetElementCount(), 1u);
  EXPECT_TRUE(view.value().GetTensorTypeAndShapeInfo().GetShape().empty());

  auto empty = allocate_tensor_msg(
    {2, 0, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  EXPECT_EQ(empty->data.size(), 0u);
  EXPECT_FALSE(static_cast<bool>(from_output_tensor_msg(*empty)));
  EXPECT_FALSE(static_cast<bool>(from_input_tensor_msg(*empty)));
}

TEST(OnnxRuntimeConversions, RejectsNonContiguousStrides)
{
  auto msg = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  msg->strides = {1, 2};

  EXPECT_THROW(from_output_tensor_msg(*msg), std::invalid_argument);
}

TEST(OnnxRuntimeConversions, RejectsOutOfBoundsView)
{
  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  msg->byte_offset = sizeof(float);

  EXPECT_THROW(from_output_tensor_msg(*msg), std::runtime_error);
}

TEST(OnnxRuntimeConversions, RejectsDtypesOnnxRuntimeCannotRepresent)
{
  EXPECT_THROW(
    allocate_tensor_msg({2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "cpu"),
    std::invalid_argument);

  // Sized so the view still fits, leaving the lane count as the only problem.
  auto msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  msg->shape = {2};
  msg->strides = {1};
  msg->dtype_lanes = 2;
  EXPECT_THROW(from_output_tensor_msg(*msg), std::invalid_argument);
}

TEST(OnnxRuntimeConversions, CopiesHostOrtValueIntoNewAndExistingMessages)
{
  std::vector<float> source{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<int64_t> shape{2, 3};
  auto value = host_value(source, shape);

  auto created = to_tensor_msg(value);
  auto existing = allocate_tensor_msg(
    {16}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  to_tensor_msg(*existing, value);

  for (const auto * msg : {created.get(), existing.get()}) {
    EXPECT_EQ(msg->shape, shape);
    EXPECT_EQ(msg->strides, (std::vector<int64_t>{3, 1}));
    EXPECT_EQ(msg->dtype_code, 2);
    EXPECT_EQ(msg->dtype_bits, 32);
    EXPECT_EQ(msg->byte_offset, 0u);
    EXPECT_EQ(msg->data.get_backend_type(), "cpu");
    const auto * result = reinterpret_cast<const float *>(msg->data.data());
    EXPECT_EQ(result[0], 1.0F);
    EXPECT_EQ(result[5], 6.0F);
  }
}

TEST(OnnxRuntimeConversions, RejectsOrtValuesLargerThanTheDestination)
{
  std::vector<float> source{1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<int64_t> shape{4};
  auto value = host_value(source, shape);
  auto msg = allocate_tensor_msg(
    {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");

  EXPECT_THROW(to_tensor_msg(*msg, value), std::runtime_error);
}

TEST(OnnxRuntimeConversions, RejectsShapeArithmeticOverflow)
{
  EXPECT_THROW(
    allocate_tensor_msg(
      {std::numeric_limits<int64_t>::max(), 2},
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "cpu"),
    std::overflow_error);
}

TEST(OnnxRuntimeConversions, RunsInferenceWithPreallocatedMessageBuffers)
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_conversions_test");
  Ort::SessionOptions session_options;
  configure_session_options(session_options, "cpu");
  Ort::Session session(
    env, identity_model, sizeof(identity_model), session_options);
  Ort::IoBinding binding(session);

  auto input = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  auto output = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
  auto * input_data = reinterpret_cast<float *>(input->data.data());
  for (size_t index = 0; index < 6; ++index) {
    input_data[index] = static_cast<float>(index + 1);
  }

  auto input_view = from_input_tensor_msg(*input);
  auto output_view = from_output_tensor_msg(*output);
  binding.BindInput("input", input_view.value());
  binding.BindOutput("output", output_view.value());
  session.Run(Ort::RunOptions{}, binding);

  const auto * output_data =
    reinterpret_cast<const float *>(output->data.data());
  for (size_t index = 0; index < 6; ++index) {
    EXPECT_EQ(output_data[index], input_data[index]);
  }
}

TEST(OnnxRuntimeConversions, ValidatesSessionProviderArguments)
{
  Ort::SessionOptions session_options;

  EXPECT_NO_THROW(configure_session_options(session_options, "cpu"));
  EXPECT_THROW(
    configure_session_options(
      session_options, "cpu", 0, reinterpret_cast<void *>(1)),
    std::invalid_argument);
  EXPECT_THROW(
    configure_session_options(session_options, "cuda"), std::invalid_argument);
  EXPECT_THROW(
    configure_session_options(
      session_options, "trainium", 0, reinterpret_cast<void *>(1)),
    std::invalid_argument);
}

}  // namespace
