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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "dlpack_conversions/dlpack_conversions.hpp"

namespace
{

constexpr DLDataType kFloat32{kDLFloat, 32, 1};
constexpr DLDataType kUint8{kDLUInt, 8, 1};

std::vector<int64_t> contiguous(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (size_t index = shape.size(); index > 0; --index) {
    strides[index - 1] = stride;
    stride *= shape[index - 1];
  }
  return strides;
}

}  // namespace

TEST(DlpackConversions, HostStorageIsAlwaysAvailable)
{
  const auto backends = dlpack_conversions::available_backends();

  EXPECT_TRUE(dlpack_conversions::backend_available("cpu"));
  EXPECT_NE(std::find(backends.begin(), backends.end(), "cpu"), backends.end());
  EXPECT_TRUE(std::is_sorted(backends.begin(), backends.end()));
  EXPECT_FALSE(dlpack_conversions::default_backend().empty());
}

TEST(DlpackConversions, HostBackendServesTheCpuDeviceType)
{
  EXPECT_EQ(dlpack_conversions::backend_for_device(kDLCPU), "cpu");
}

TEST(DlpackConversions, UnknownDeviceTypesHaveNoBackend)
{
  EXPECT_TRUE(dlpack_conversions::backend_for_device(kDLVulkan).empty());
}

TEST(DlpackConversions, UninstalledBackendsAreRejected)
{
  EXPECT_FALSE(dlpack_conversions::backend_available("no_such_backend"));
  EXPECT_THROW(
    dlpack_conversions::allocate_tensor_msg({2}, kFloat32, "no_such_backend"),
    std::runtime_error);
}

TEST(DlpackConversions, AllocationStampsContiguousMetadata)
{
  const auto msg = dlpack_conversions::allocate_tensor_msg(
    {2, 3, 4}, kFloat32, "cpu");

  EXPECT_EQ(std::vector<int64_t>(msg->shape.begin(), msg->shape.end()),
    std::vector<int64_t>({2, 3, 4}));
  EXPECT_EQ(std::vector<int64_t>(msg->strides.begin(), msg->strides.end()),
    std::vector<int64_t>({12, 4, 1}));
  EXPECT_EQ(msg->dtype_code, kDLFloat);
  EXPECT_EQ(msg->dtype_bits, 32);
  EXPECT_EQ(msg->dtype_lanes, 1);
  EXPECT_EQ(msg->byte_offset, 0u);
  EXPECT_EQ(msg->data.size(), 2u * 3u * 4u * 4u);
}

TEST(DlpackConversions, ViewsAliasMessageStorageWithoutCopying)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cpu");

  const auto output = dlpack_conversions::from_output_tensor_msg(*msg, 0);
  const auto input = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  ASSERT_TRUE(output);
  ASSERT_TRUE(input);
  EXPECT_EQ(output.get()->dl_tensor.data, input.get()->dl_tensor.data);
  EXPECT_EQ(output.get()->dl_tensor.data, msg->data.data());
  EXPECT_EQ(output.get()->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(output.get()->dl_tensor.ndim, 1);
  EXPECT_EQ(output.get()->dl_tensor.byte_offset, 0u);
}

TEST(DlpackConversions, AByteOffsetIsFoldedIntoThePointer)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({8}, kUint8, "cpu");
  const auto base = dlpack_conversions::from_input_tensor_msg(
    *msg, 0).get()->dl_tensor.data;
  msg->shape = {4};
  msg->strides = {1};
  msg->byte_offset = 4;

  const auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  EXPECT_EQ(
    static_cast<uint8_t *>(view.get()->dl_tensor.data),
    static_cast<uint8_t *>(base) + 4);
  EXPECT_EQ(view.get()->dl_tensor.byte_offset, 0u);
}

TEST(DlpackConversions, AnEmptyMessageYieldsNoTensor)
{
  dlpack_conversions::TensorMsg msg;

  EXPECT_FALSE(dlpack_conversions::from_input_tensor_msg(msg, 0));
  EXPECT_FALSE(dlpack_conversions::from_output_tensor_msg(msg, 0));
}

TEST(DlpackConversions, ZeroSizedShapesAreAccepted)
{
  const auto msg = dlpack_conversions::allocate_tensor_msg(
    {0, 3}, kFloat32, "cpu");

  EXPECT_EQ(msg->data.size(), 0u);
  EXPECT_FALSE(dlpack_conversions::from_input_tensor_msg(*msg, 0));
}

TEST(DlpackConversions, ScalarShapesAreAccepted)
{
  const auto msg = dlpack_conversions::allocate_tensor_msg({}, kFloat32, "cpu");

  const auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  ASSERT_TRUE(view);
  EXPECT_EQ(view.get()->dl_tensor.ndim, 0);
  EXPECT_EQ(msg->data.size(), 4u);
}

TEST(DlpackConversions, ReleasingHandsOffTheDeleter)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cpu");
  auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  DLManagedTensor * raw = view.release();

  ASSERT_NE(raw, nullptr);
  EXPECT_FALSE(view);
  ASSERT_NE(raw->deleter, nullptr);
  raw->deleter(raw);
}

TEST(DlpackConversions, MovingTransfersTheLease)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cpu");
  auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);
  const auto * expected = view.get();

  const dlpack_conversions::ManagedTensor moved(std::move(view));

  EXPECT_EQ(moved.get(), expected);
  EXPECT_FALSE(view);
}

// The core is the last line of defence before a framework receives a raw
// pointer, so a message whose view reaches past its own storage must be
// rejected rather than trusted.
TEST(DlpackConversions, ViewsReachingPastTheirStorageAreRejected)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cpu");
  msg->shape = {5};
  msg->strides = {1};

  EXPECT_THROW(
    dlpack_conversions::from_input_tensor_msg(*msg, 0), std::runtime_error);
  EXPECT_THROW(
    dlpack_conversions::from_output_tensor_msg(*msg, 0), std::runtime_error);
}

TEST(DlpackConversions, AByteOffsetPastTheEndIsRejected)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cpu");
  msg->byte_offset = 4;

  EXPECT_THROW(
    dlpack_conversions::from_input_tensor_msg(*msg, 0), std::runtime_error);
}

TEST(DlpackConversions, OverlongStridesAreRejected)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({2, 2}, kFloat32, "cpu");
  msg->strides = {8, 1};

  EXPECT_THROW(
    dlpack_conversions::from_input_tensor_msg(*msg, 0), std::runtime_error);
}

TEST(DlpackConversions, StridesMustMatchTheShapeRank)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({2, 2}, kFloat32, "cpu");
  msg->strides = {1};

  EXPECT_THROW(
    dlpack_conversions::from_input_tensor_msg(*msg, 0), std::runtime_error);
}

TEST(DlpackConversions, EmptyStridesFallBackToRowMajor)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({2, 3}, kFloat32, "cpu");
  msg->strides.clear();

  const auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  ASSERT_TRUE(view);
  EXPECT_EQ(view.get()->dl_tensor.strides, nullptr);
}

TEST(DlpackConversions, CopyingStampsMetadataAndMovesBytes)
{
  std::vector<float> source(6);
  std::iota(source.begin(), source.end(), 1.0f);
  std::vector<int64_t> shape{2, 3};
  auto strides = contiguous(shape);
  DLTensor tensor{};
  tensor.data = source.data();
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 2;
  tensor.dtype = kFloat32;
  tensor.shape = shape.data();
  tensor.strides = strides.data();

  const auto msg = dlpack_conversions::to_tensor_msg(tensor, 0);

  ASSERT_EQ(msg->data.size(), source.size() * sizeof(float));
  EXPECT_EQ(std::vector<int64_t>(msg->shape.begin(), msg->shape.end()), shape);
  EXPECT_EQ(std::vector<int64_t>(msg->strides.begin(), msg->strides.end()),
    strides);
  EXPECT_EQ(msg->dtype_code, kDLFloat);
  EXPECT_EQ(msg->byte_offset, 0u);
  const auto * copied = reinterpret_cast<const float *>(msg->data.data());
  EXPECT_TRUE(std::equal(source.begin(), source.end(), copied));
}

TEST(DlpackConversions, CopyingIntoATooSmallMessageIsRejected)
{
  std::vector<float> source(6, 1.0f);
  std::vector<int64_t> shape{6};
  auto strides = contiguous(shape);
  DLTensor tensor{};
  tensor.data = source.data();
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = kFloat32;
  tensor.shape = shape.data();
  tensor.strides = strides.data();
  auto msg = dlpack_conversions::allocate_tensor_msg({2}, kFloat32, "cpu");

  EXPECT_THROW(
    dlpack_conversions::to_tensor_msg(*msg, tensor, 0), std::runtime_error);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
