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
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "torch_conversions/torch_conversions.hpp"

TEST(TorchConversions, AllocatePopulatesMetadata)
{
  auto msg = torch_conversions::allocate_tensor_msg(
    {2, 3, 4}, at::kFloat, c10::kCPU);

  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{12, 4, 1}));
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(2));
  EXPECT_EQ(msg->dtype_bits, 32u);
  EXPECT_EQ(msg->dtype_lanes, 1u);
  EXPECT_EQ(msg->byte_offset, 0u);
  EXPECT_EQ(msg->data.size(), 24u * sizeof(float));
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");
}

TEST(TorchConversions, RoundTrip)
{
  auto source = torch::arange(12, torch::kFloat).reshape({3, 4});
  auto msg = torch_conversions::to_tensor_msg(source);
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");
  EXPECT_TRUE(torch::equal(
      source,
      torch_conversions::from_input_tensor_msg(*msg)));
}

TEST(TorchConversions, OutputViewAliasesMessageStorage)
{
  auto msg = torch_conversions::allocate_tensor_msg(
    {4}, at::kInt, c10::kCPU);
  auto output = torch_conversions::from_output_tensor_msg(*msg);
  output.copy_(torch::tensor({10, 20, 30, 40}, at::kInt));

  auto input = torch_conversions::from_input_tensor_msg(*msg, false);
  EXPECT_EQ(
    input.data_ptr<int32_t>(),
    reinterpret_cast<int32_t *>(msg->data.data()));
  EXPECT_TRUE(torch::equal(input, output));
}

TEST(TorchConversions, ByteOffsetSelectsStorageSubview)
{
  auto msg = torch_conversions::allocate_tensor_msg(
    {8}, at::kInt, c10::kCPU);
  auto full = torch_conversions::from_output_tensor_msg(*msg);
  full.copy_(torch::arange(8, at::kInt));
  msg->shape = {3};
  msg->strides = {1};
  msg->byte_offset = 2 * sizeof(int32_t);

  auto view = torch_conversions::from_input_tensor_msg(*msg, false);
  EXPECT_TRUE(torch::equal(view, torch::tensor({2, 3, 4}, at::kInt)));
}

TEST(TorchConversions, ToExistingMessageUpdatesMetadata)
{
  auto msg = torch_conversions::allocate_tensor_msg(
    {16}, at::kFloat, c10::kCPU);
  auto source = torch::arange(6, at::kFloat).reshape({2, 3});
  torch_conversions::to_tensor_msg(*msg, source);

  EXPECT_EQ(msg->shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(msg->strides, (std::vector<int64_t>{3, 1}));
  EXPECT_TRUE(torch::equal(
      source, torch_conversions::from_input_tensor_msg(*msg)));
}

TEST(TorchConversions, RejectsOversizedTensor)
{
  auto msg = torch_conversions::allocate_tensor_msg(
    {4}, at::kByte, c10::kCPU);
  EXPECT_THROW(
    torch_conversions::to_tensor_msg(*msg, torch::zeros({128}, at::kByte)),
    std::runtime_error);
}

TEST(TorchConversions, RejectsShapeArithmeticOverflow)
{
  EXPECT_THROW(
    torch_conversions::allocate_tensor_msg(
      {std::numeric_limits<int64_t>::max(), 2}, at::kByte, c10::kCPU),
    std::overflow_error);
}

TEST(TorchConversions, EmptyDataReturnsUndefinedTensor)
{
  torch_conversions::TensorMsg msg;
  EXPECT_FALSE(torch_conversions::from_input_tensor_msg(msg).defined());
  EXPECT_FALSE(torch_conversions::from_output_tensor_msg(msg).defined());
}

TEST(TorchConversions, RejectsDeviceWithoutStoragePlugin)
{
  if (torch_conversions::backend_available("cuda")) {
    GTEST_SKIP() << "the CUDA storage plugin is installed";
  }
  EXPECT_THROW(
    torch_conversions::allocate_tensor_msg(
      {1}, at::kFloat, c10::kCUDA),
    std::runtime_error);
}

TEST(TorchConversions, ReportsInstalledBackends)
{
  const auto backends = torch_conversions::available_backends();
  EXPECT_NE(
    std::find(backends.begin(), backends.end(), "cpu"), backends.end());
  EXPECT_FALSE(torch_conversions::default_backend().empty());
}

// An accelerator storage plugin can be installed next to a LibTorch built
// without kernels for that device, so an allocation that names no device has
// to stay usable rather than abort inside ATen on first touch.
TEST(TorchConversions, DefaultAllocationsAreUsableByThisTorchBuild)
{
  auto msg = torch_conversions::allocate_tensor_msg({4}, at::kFloat);

  at::Tensor tensor = torch_conversions::from_output_tensor_msg(*msg);

  ASSERT_TRUE(tensor.defined());
  EXPECT_NO_THROW(tensor.fill_(1.0f));
}
