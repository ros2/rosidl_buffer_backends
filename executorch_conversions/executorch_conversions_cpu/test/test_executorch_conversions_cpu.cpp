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

#include <gtest/gtest.h>
#include <executorch/extension/module/module.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "executorch_conversions/executorch_conversions.hpp"

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;
using executorch::extension::make_tensor_ptr;

namespace
{

template<typename T>
std::vector<T> values(const executorch::aten::Tensor & tensor)
{
  const auto * data = tensor.const_data_ptr<T>();
  return {data, data + tensor.numel()};
}

const Device kCpu(DeviceType::CPU);

}  // namespace

TEST(ExecuTorchConversions, AllocatePopulatesMetadata)
{
  auto msg = executorch_conversions::allocate_tensor_msg({2, 3, 4}, ScalarType::Float, kCpu);

  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{12, 4, 1}));
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(2));
  EXPECT_EQ(msg->dtype_bits, 32u);
  EXPECT_EQ(msg->dtype_lanes, 1u);
  EXPECT_EQ(msg->byte_offset, 0u);
  EXPECT_EQ(msg->data.size(), 24u * sizeof(float));
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");
}

TEST(ExecuTorchConversions, RoundTrip)
{
  auto source = make_tensor_ptr({3, 4}, std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  auto msg = executorch_conversions::to_tensor_msg(*source);
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");

  auto view = executorch_conversions::from_input_tensor_msg(*msg);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->scalar_type(), ScalarType::Float);
  EXPECT_EQ(view->device().type(), DeviceType::CPU);
  EXPECT_EQ(std::vector<int32_t>(view->sizes().begin(), view->sizes().end()),
    (std::vector<int32_t>{3, 4}));
  EXPECT_EQ(values<float>(*view), values<float>(*source));
}

TEST(ExecuTorchConversions, OutputViewAliasesMessageStorage)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCpu);
  auto output = executorch_conversions::from_output_tensor_msg(*msg);
  const int32_t source[] = {10, 20, 30, 40};
  std::memcpy(output->mutable_data_ptr<int32_t>(), source, sizeof(source));

  auto input = executorch_conversions::from_input_tensor_msg(*msg);
  EXPECT_EQ(
    input->const_data_ptr<int32_t>(),
    reinterpret_cast<const int32_t *>(msg->data.data()));
  EXPECT_EQ(values<int32_t>(*input), (std::vector<int32_t>{10, 20, 30, 40}));
}

TEST(ExecuTorchConversions, ByteOffsetSelectsStorageSubview)
{
  auto msg = executorch_conversions::allocate_tensor_msg({8}, ScalarType::Int, kCpu);
  auto full = executorch_conversions::from_output_tensor_msg(*msg);
  for (int32_t index = 0; index < 8; ++index) {
    full->mutable_data_ptr<int32_t>()[index] = index;
  }
  msg->shape = {3};
  msg->strides = {1};
  msg->byte_offset = 2 * sizeof(int32_t);

  auto view = executorch_conversions::from_input_tensor_msg(*msg);
  EXPECT_EQ(values<int32_t>(*view), (std::vector<int32_t>{2, 3, 4}));
}

TEST(ExecuTorchConversions, ToExistingMessageUpdatesMetadata)
{
  auto msg = executorch_conversions::allocate_tensor_msg({16}, ScalarType::Float, kCpu);
  auto source = make_tensor_ptr({2, 3}, std::vector<float>{0, 1, 2, 3, 4, 5});
  executorch_conversions::to_tensor_msg(*msg, *source);

  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{3, 1}));
  EXPECT_EQ(
    values<float>(*executorch_conversions::from_input_tensor_msg(*msg)),
    values<float>(*source));
}

TEST(ExecuTorchConversions, RejectsPermutedLayouts)
{
  // NHWC storage of an NCHW tensor. A Method would read its bytes as NCHW.
  auto source = make_tensor_ptr(
    {1, 2, 2, 2}, std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 3, 1}, {});
  EXPECT_THROW(executorch_conversions::to_tensor_msg(*source), std::invalid_argument);
  auto msg = executorch_conversions::allocate_tensor_msg({1, 2, 2, 2}, ScalarType::Float, kCpu);
  msg->strides = {8, 1, 4, 2};
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::runtime_error);
  EXPECT_THROW(executorch_conversions::from_output_tensor_msg(*msg), std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsStridesExecuTorchCannotView)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4, 4}, ScalarType::Float, kCpu);
  msg->shape = {2, 2};
  msg->strides = {8, 2};  // every other element: not dense
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsShapesBeyondExecuTorchLimits)
{
  executorch_conversions::TensorMsg msg;
  msg.dtype_code = 1;
  msg.dtype_bits = 8;
  msg.dtype_lanes = 1;
  msg.shape = {int64_t{1} << 31, 0};
  msg.data.resize(1);
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(msg), std::overflow_error);
  msg.shape = {0, 65536, 65536};  // the outer stride is 2^32
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(msg), std::overflow_error);
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg({int64_t{1} << 31}, ScalarType::Byte, kCpu),
    std::overflow_error);
}

TEST(ExecuTorchConversions, CountsSizeZeroDimensionsAsOneInStrides)
{
  EXPECT_EQ(
    executorch_conversions::allocate_tensor_msg({2, 0, 3}, ScalarType::Float, kCpu)->strides,
    (std::vector<int64_t>{3, 3, 1}));
  executorch_conversions::TensorMsg msg;
  msg.dtype_code = 1;
  msg.dtype_bits = 8;
  msg.dtype_lanes = 1;
  msg.shape = {2, 0, 3};
  msg.strides = {3, 3, 1};
  msg.data.resize(1);
  EXPECT_EQ(executorch_conversions::from_input_tensor_msg(msg)->numel(), 0);
  msg.shape = {2, 65536, 0, 65536};  // the outer stride is 2^32
  msg.strides.clear();
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(msg), std::overflow_error);
}

TEST(ExecuTorchConversions, RejectsRanksBeyondExecuTorchLimits)
{
  executorch_conversions::TensorMsg msg;
  msg.dtype_code = 1;
  msg.dtype_bits = 8;
  msg.dtype_lanes = 1;
  msg.shape.assign(17, 1);
  msg.data.resize(1);
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(msg), std::runtime_error);
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg(msg.shape, ScalarType::Byte, kCpu),
    std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsOversizedTensor)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Byte, kCpu);
  auto source = make_tensor_ptr({128}, std::vector<uint8_t>(128), {}, {}, ScalarType::Byte);
  EXPECT_THROW(executorch_conversions::to_tensor_msg(*msg, *source), std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsViewBeyondStorage)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  msg->byte_offset = 4;
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsMisalignedByteOffset)
{
  auto msg = executorch_conversions::allocate_tensor_msg({8}, ScalarType::Float, kCpu);
  msg->shape = {4};
  msg->strides = {1};
  msg->byte_offset = 2;
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsATensorWithoutData)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  auto empty = make_tensor_ptr({4}, static_cast<void *>(nullptr), ScalarType::Float);
  EXPECT_THROW(executorch_conversions::to_tensor_msg(*msg, *empty), std::invalid_argument);
}

TEST(ExecuTorchConversions, RejectsATensorWithoutStrides)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  std::vector<float> data(4, 1.0f);
  executorch::aten::SizesType sizes[] = {4};
  executorch::aten::TensorImpl impl(ScalarType::Float, 1, sizes, data.data());
  EXPECT_THROW(
    executorch_conversions::to_tensor_msg(*msg, executorch::aten::Tensor(&impl)),
    std::invalid_argument);
}

TEST(ExecuTorchConversions, WritesContiguousStridesForSizeOneDimensions)
{
  std::vector<float> data{1, 2, 3};
  executorch::aten::SizesType sizes[] = {1, 3};
  executorch::aten::DimOrderType dim_order[] = {0, 1};
  executorch::aten::StridesType strides[] = {-5, 1};
  executorch::aten::TensorImpl impl(ScalarType::Float, 2, sizes, data.data(), dim_order, strides);
  auto msg = executorch_conversions::to_tensor_msg(executorch::aten::Tensor(&impl));
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{3, 1}));
  EXPECT_EQ(values<float>(*executorch_conversions::from_input_tensor_msg(*msg)), data);
}

TEST(ExecuTorchConversions, ConvertsARankZeroTensor)
{
  auto source = make_tensor_ptr(std::vector<executorch::aten::SizesType>{}, std::vector<float>{5});
  auto msg = executorch_conversions::to_tensor_msg(*source);
  EXPECT_TRUE(msg->shape.empty());
  auto view = executorch_conversions::from_input_tensor_msg(*msg);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->dim(), 0);
  auto copy = executorch_conversions::to_tensor_msg(*view);
  EXPECT_EQ(values<float>(*executorch_conversions::from_input_tensor_msg(*copy)),
    std::vector<float>{5});
}

TEST(ExecuTorchConversions, RejectsAByteOffsetThatWrapsAround)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  msg->byte_offset = std::numeric_limits<uint64_t>::max() - 7;
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::overflow_error);
}

TEST(ExecuTorchConversions, RejectsASpanBeyondSizeLimits)
{
  auto msg = executorch_conversions::allocate_tensor_msg({537552}, ScalarType::Byte, kCpu);
  msg->dtype_code = 0;
  msg->dtype_bits = 64;
  msg->shape = {2147437309, 1073764994};
  msg->strides.clear();
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::overflow_error);
}

TEST(ExecuTorchConversions, CopyTakesTheTensorDtype)
{
  auto msg = executorch_conversions::allocate_tensor_msg({16}, ScalarType::Byte, kCpu);
  executorch_conversions::to_tensor_msg(*msg,
    *make_tensor_ptr({4}, std::vector<float>{1, 2, 3, 4}));
  EXPECT_EQ(executorch_conversions::scalar_type(*msg), ScalarType::Float);
  EXPECT_EQ(
    values<float>(*executorch_conversions::from_input_tensor_msg(*msg)),
    (std::vector<float>{1, 2, 3, 4}));
}

TEST(ExecuTorchConversions, RejectsNegativeShapes)
{
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg({-1, 0}, ScalarType::Float, kCpu),
    std::runtime_error);
  auto msg = executorch_conversions::allocate_tensor_msg({1}, ScalarType::Float, kCpu);
  msg->shape = {-1, 0};
  msg->strides.clear();
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::runtime_error);
}

TEST(ExecuTorchConversions, ViewsAMessageWithoutStrides)
{
  auto msg = executorch_conversions::to_tensor_msg(
    *make_tensor_ptr({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}));
  msg->strides.clear();
  EXPECT_EQ(
    values<float>(*executorch_conversions::from_input_tensor_msg(*msg)),
    (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(ExecuTorchConversions, CopiesAViewIntoItsOwnMessage)
{
  auto msg = executorch_conversions::allocate_tensor_msg({8}, ScalarType::Int, kCpu);
  for (int32_t i = 0; i < 8; ++i) {
    reinterpret_cast<int32_t *>(msg->data.data())[i] = i;
  }
  msg->shape = {6};
  msg->byte_offset = 8;
  auto view = executorch_conversions::from_input_tensor_msg(*msg);
  executorch_conversions::to_tensor_msg(*msg, *view);
  EXPECT_EQ(msg->byte_offset, 0u);
  const auto * values = reinterpret_cast<const int32_t *>(msg->data.data());
  EXPECT_EQ(std::vector<int32_t>(values, values + 6), (std::vector<int32_t>{2, 3, 4, 5, 6, 7}));
}

TEST(ExecuTorchConversions, RejectsOutputViewBeyondStorage)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  msg->byte_offset = 4;
  EXPECT_THROW(executorch_conversions::from_output_tensor_msg(*msg), std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsMultidimensionalViewBeyondStorage)
{
  executorch_conversions::TensorMsg msg;
  msg.dtype_code = 2;
  msg.dtype_bits = 32;
  msg.dtype_lanes = 1;
  msg.shape = {2, 3};
  msg.data.resize(5 * sizeof(float));
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(msg), std::runtime_error);
  EXPECT_THROW(executorch_conversions::from_output_tensor_msg(msg), std::runtime_error);
  msg.data.resize(6 * sizeof(float) - 1);
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(msg), std::runtime_error);
  EXPECT_THROW(executorch_conversions::from_output_tensor_msg(msg), std::runtime_error);
}

TEST(ExecuTorchConversions, UsesDLPackDtypeCodes)
{
  struct Case
  {
    ScalarType dtype;
    uint8_t code;
    uint8_t bits;
  };
  const Case cases[] = {
    {ScalarType::Byte, 1, 8}, {ScalarType::Char, 0, 8}, {ScalarType::Short, 0, 16},
    {ScalarType::Int, 0, 32}, {ScalarType::Long, 0, 64}, {ScalarType::Half, 2, 16},
    {ScalarType::Float, 2, 32}, {ScalarType::Double, 2, 64},
    {ScalarType::BFloat16, 4, 16}, {ScalarType::Bool, 6, 8}};
  for (const auto & c : cases) {
    SCOPED_TRACE(static_cast<int>(c.dtype));
    auto msg = executorch_conversions::allocate_tensor_msg({2}, c.dtype, kCpu);
    EXPECT_EQ(msg->dtype_code, c.code);
    EXPECT_EQ(msg->dtype_bits, c.bits);
    EXPECT_EQ(msg->dtype_lanes, 1u);
    EXPECT_EQ(executorch_conversions::scalar_type(*msg), c.dtype);
  }
}

TEST(ExecuTorchConversions, RoundTripsTheBytesOfEveryDtype)
{
  const std::pair<ScalarType, size_t> cases[] = {
    {ScalarType::Byte, 1}, {ScalarType::Char, 1}, {ScalarType::Short, 2}, {ScalarType::Int, 4},
    {ScalarType::Long, 8}, {ScalarType::Half, 2}, {ScalarType::Float, 4},
    {ScalarType::Double, 8}, {ScalarType::BFloat16, 2}, {ScalarType::Bool, 1}};
  for (const auto & [dtype, width] : cases) {
    SCOPED_TRACE(static_cast<int>(dtype));
    std::vector<uint8_t> bytes(3 * width);
    for (size_t i = 0; i < bytes.size(); ++i) {
      // A Bool element must be 0 or 1.
      bytes[i] = static_cast<uint8_t>(dtype == ScalarType::Bool ? i % 2 : i + 1);
    }
    auto source = make_tensor_ptr({3}, bytes.data(), {}, {}, dtype);
    auto msg = executorch_conversions::to_tensor_msg(*source);
    ASSERT_EQ(msg->data.size(), bytes.size());
    EXPECT_EQ(std::memcmp(msg->data.data(), bytes.data(), bytes.size()), 0);
    auto view = executorch_conversions::from_input_tensor_msg(*msg);
    ASSERT_EQ(view->nbytes(), bytes.size());
    EXPECT_EQ(std::memcmp(view->const_data_ptr(), bytes.data(), bytes.size()), 0);
  }
}

TEST(ExecuTorchConversions, RejectsUnsupportedDtypeWidths)
{
  executorch_conversions::TensorMsg msg;
  msg.dtype_lanes = 1;
  const std::pair<uint8_t, uint8_t> cases[] = {{1, 16}, {4, 32}, {6, 1}, {6, 16}};
  for (const auto & [code, bits] : cases) {
    SCOPED_TRACE(std::to_string(code) + ":" + std::to_string(bits));
    msg.dtype_code = code;
    msg.dtype_bits = bits;
    EXPECT_THROW(executorch_conversions::scalar_type(msg), std::runtime_error);
  }
}

TEST(ExecuTorchConversions, ReportsInstalledBackends)
{
  const auto backends = executorch_conversions::available_backends();
  EXPECT_NE(std::find(backends.begin(), backends.end(), "cpu"), backends.end());
  EXPECT_FALSE(executorch_conversions::default_backend().empty());
}

TEST(ExecuTorchConversions, RejectsShapeArithmeticOverflow)
{
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg(
      {std::numeric_limits<int64_t>::max(), 2}, ScalarType::Byte, kCpu),
    std::overflow_error);
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg({int64_t{1} << 60, 4}, ScalarType::Long, kCpu),
    std::overflow_error);
  // Sizes and strides fit in 32 bits, but the byte count does not fit in size_t.
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg({2147483647, 2147483647}, ScalarType::Long, kCpu),
    std::overflow_error);
}

TEST(ExecuTorchConversions, RejectsStridesOfAnotherRank)
{
  auto msg = executorch_conversions::allocate_tensor_msg({2, 3}, ScalarType::Float, kCpu);
  // The first 2 strides alone would describe a contiguous tensor.
  msg->strides = {3, 1, 1};
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsNegativeStrides)
{
  auto msg = executorch_conversions::allocate_tensor_msg({1, 3}, ScalarType::Float, kCpu);
  msg->strides = {-3, 1};
  EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::runtime_error);
}

TEST(ExecuTorchConversions, EmptyTensorLeavesTheMessageUnchanged)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  auto source = make_tensor_ptr({4}, std::vector<float>{1, 2, 3, 4});
  executorch_conversions::to_tensor_msg(*msg, *source);
  auto empty = make_tensor_ptr({0}, std::vector<double>{});
  executorch_conversions::to_tensor_msg(*msg, *empty);
  EXPECT_EQ(msg->shape, (std::vector<int64_t>{4}));
  EXPECT_EQ(executorch_conversions::scalar_type(*msg), ScalarType::Float);
  EXPECT_EQ(
    values<float>(*executorch_conversions::from_input_tensor_msg(*msg)),
    (std::vector<float>{1, 2, 3, 4}));
}

TEST(ExecuTorchConversions, EmptyTensorGivesAnEmptyMessage)
{
  auto empty = make_tensor_ptr({0}, std::vector<float>{});
  auto msg = executorch_conversions::to_tensor_msg(*empty);
  EXPECT_TRUE(msg->shape.empty());
  EXPECT_TRUE(msg->data.empty());
}

TEST(ExecuTorchConversions, AcceptsAnyStrideOnSizeOneDimensions)
{
  auto source = make_tensor_ptr({1, 4}, std::vector<float>{0, 1, 2, 3});
  auto msg = executorch_conversions::to_tensor_msg(*source);
  msg->strides = {99, 1};
  EXPECT_EQ(
    values<float>(*executorch_conversions::from_input_tensor_msg(*msg)),
    (std::vector<float>{0, 1, 2, 3}));
}

TEST(ExecuTorchConversions, UnsupportedDtypeLeavesTheMessageUnchanged)
{
  auto source = make_tensor_ptr({2, 2}, std::vector<float>{0, 1, 2, 3});
  auto msg = executorch_conversions::to_tensor_msg(*source);
  auto complex = make_tensor_ptr(
    {2}, std::vector<uint8_t>(2 * 8, 0xff), {}, {}, ScalarType::ComplexFloat);
  EXPECT_THROW(executorch_conversions::to_tensor_msg(*msg, *complex), std::runtime_error);
  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(
    values<float>(*executorch_conversions::from_input_tensor_msg(*msg)),
    (std::vector<float>{0, 1, 2, 3}));
}

TEST(ExecuTorchConversions, DefaultAllocationsRoundTrip)
{
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float);
  auto source = make_tensor_ptr({4}, std::vector<float>{3, 3, 3, 3});
  executorch_conversions::to_tensor_msg(*msg, *source);

  auto view = executorch_conversions::from_input_tensor_msg(*msg);
  auto host = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  executorch_conversions::to_tensor_msg(*host, *view);
  EXPECT_EQ(
    values<float>(*executorch_conversions::from_input_tensor_msg(*host)),
    (std::vector<float>{3, 3, 3, 3}));
}

TEST(ExecuTorchConversions, ModuleReadsAndWritesMessageStorage)
{
  executorch::extension::Module module(ADD_ONE_PTE);
  const auto method_meta = module.method_meta("forward");
  ASSERT_TRUE(method_meta.ok());
  const auto input_meta = method_meta->input_tensor_meta(0);
  ASSERT_TRUE(input_meta.ok());
  // The runtime copies a planned input, so the program would not read message storage.
  EXPECT_FALSE(input_meta->is_memory_planned());
  auto request = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  executorch_conversions::to_tensor_msg(
    *request, *make_tensor_ptr({4}, std::vector<float>{1, 2, 3, 4}));
  auto reply = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCpu);
  {
    auto input = executorch_conversions::from_input_tensor_msg(*request);
    auto output = executorch_conversions::from_output_tensor_msg(*reply);
    ASSERT_EQ(module.set_output(*output), executorch::runtime::Error::Ok);
    const auto result = module.forward(*input);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->at(0).toTensor().const_data_ptr(), output->const_data_ptr());
  }
  EXPECT_EQ(
    values<float>(*executorch_conversions::from_input_tensor_msg(*reply)),
    (std::vector<float>{2, 3, 4, 5}));
}
