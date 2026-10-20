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

#include <vector>

#include "torch_conversions/torch_conversions.hpp"

#ifdef TORCH_CONVERSIONS_HAS_CUDA
#include <torch/cuda.h>
#endif

using torch_conversions::TensorMsg;
using torch_conversions::ImageMsg;
using torch_conversions::allocate_image_msg;
using torch_conversions::from_input_image_msg;
using torch_conversions::from_output_image_msg;
using torch_conversions::to_image_msg;
using torch_conversions::allocate_tensor_msg;
using torch_conversions::from_input_tensor_msg;
using torch_conversions::from_output_tensor_msg;
using torch_conversions::to_tensor_msg;

// Note: kDLCPU, kDLUInt, kDLInt, kDLFloat are enumerators of DLDataTypeCode /
// DLDeviceType declared at global scope by <ATen/dlpack.h> (transitively
// included by torch_conversions.hpp), so they are already visible here.

TEST(TorchTensorBridge, AllocateCpuTensorPopulatesDlpackMetadata)
{
  auto msg = allocate_tensor_msg({2, 3, 4}, at::kFloat, c10::kCPU);

  ASSERT_EQ(msg->shape.size(), 3u);
  EXPECT_EQ(msg->shape[0], 2);
  EXPECT_EQ(msg->shape[1], 3);
  EXPECT_EQ(msg->shape[2], 4);

  ASSERT_EQ(msg->strides.size(), 3u);
  EXPECT_EQ(msg->strides[0], 12);
  EXPECT_EQ(msg->strides[1], 4);
  EXPECT_EQ(msg->strides[2], 1);

  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(msg->dtype_bits, 32u);
  EXPECT_EQ(msg->dtype_lanes, 1u);

  EXPECT_EQ(msg->byte_offset, 0u);
  EXPECT_EQ(msg->data.size(), 2u * 3u * 4u * sizeof(float));
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");
}

TEST(TorchTensorBridge, ByteDtypeRoundTripsThroughDlpackTriple)
{
  auto msg = allocate_tensor_msg({5}, at::kByte, c10::kCPU);
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLUInt));
  EXPECT_EQ(msg->dtype_bits, 8u);
  EXPECT_EQ(msg->dtype_lanes, 1u);
}

TEST(TorchTensorBridge, Int32DtypeRoundTripsThroughDlpackTriple)
{
  auto msg = allocate_tensor_msg({4}, at::kInt, c10::kCPU);
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLInt));
  EXPECT_EQ(msg->dtype_bits, 32u);
}

TEST(TorchTensorBridge, WriteThenReadRoundTrip)
{
  auto msg = allocate_tensor_msg({4}, at::kInt, c10::kCPU);

  {
    at::Tensor t = from_output_tensor_msg(*msg);
    ASSERT_TRUE(t.defined());
    EXPECT_EQ(t.sizes(), (std::vector<int64_t>{4}));
    EXPECT_EQ(t.scalar_type(), at::kInt);
    t.copy_(torch::tensor({10, 20, 30, 40}, at::kInt));
  }

  at::Tensor v = from_input_tensor_msg(
    *msg, /*clone=*/false);
  ASSERT_TRUE(v.defined());
  ASSERT_EQ(v.numel(), 4);
  auto * p = v.data_ptr<int32_t>();
  EXPECT_EQ(p[0], 10);
  EXPECT_EQ(p[1], 20);
  EXPECT_EQ(p[2], 30);
  EXPECT_EQ(p[3], 40);
}

TEST(TorchTensorBridge, ToTensorMsgCopiesAndUpdatesMetadata)
{
  auto msg = allocate_tensor_msg({16}, at::kFloat, c10::kCPU);

  at::Tensor src = torch::arange(0, 6, at::kFloat).reshape({2, 3});
  to_tensor_msg(*msg, src);

  ASSERT_EQ(msg->shape.size(), 2u);
  EXPECT_EQ(msg->shape[0], 2);
  EXPECT_EQ(msg->shape[1], 3);
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(msg->dtype_bits, 32u);
  EXPECT_EQ(msg->byte_offset, 0u);

  at::Tensor round = from_input_tensor_msg(
    *msg, /*clone=*/true);
  ASSERT_EQ(round.numel(), 6);
  EXPECT_TRUE(torch::equal(round.flatten(), src.flatten()));
}

TEST(TorchTensorBridge, ToTensorMsgAllocatesAndCopies)
{
  at::Tensor src = torch::arange(0, 6, at::kFloat).reshape({2, 3});
  auto msg = to_tensor_msg(src);

  ASSERT_EQ(msg->shape.size(), 2u);
  EXPECT_EQ(msg->shape[0], 2);
  EXPECT_EQ(msg->shape[1], 3);
  EXPECT_EQ(msg->dtype_code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(msg->dtype_bits, 32u);
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");

  at::Tensor round = from_input_tensor_msg(*msg, /*clone=*/true);
  ASSERT_EQ(round.numel(), 6);
  EXPECT_TRUE(torch::equal(round.flatten(), src.flatten()));
}

TEST(TorchTensorBridge, ByteOffsetSelectsSubregionOfStorage)
{
  // Allocate 16 ints but publish only a 4-int view starting at index 4.
  auto msg = allocate_tensor_msg({16}, at::kInt, c10::kCPU);
  {
    at::Tensor full = from_output_tensor_msg(*msg);
    for (int i = 0; i < 16; ++i) {
      full.index_put_({i}, i * 100);
    }
  }

  msg->shape = {4};
  msg->strides = {1};
  msg->byte_offset = 4 * sizeof(int32_t);

  at::Tensor view = from_input_tensor_msg(
    *msg, /*clone=*/false);
  ASSERT_EQ(view.numel(), 4);
  auto * p = view.data_ptr<int32_t>();
  EXPECT_EQ(p[0], 400);
  EXPECT_EQ(p[1], 500);
  EXPECT_EQ(p[2], 600);
  EXPECT_EQ(p[3], 700);
}

TEST(TorchTensorBridge, ToTensorMsgRejectsOversizedTensor)
{
  auto msg = allocate_tensor_msg({4}, at::kByte, c10::kCPU);
  at::Tensor big = torch::zeros({128}, at::kByte);
  EXPECT_THROW(to_tensor_msg(*msg, big), std::runtime_error);
}

TEST(TorchTensorBridge, EmptyDataReturnsUndefinedTensor)
{
  TensorMsg msg;
  EXPECT_FALSE(from_input_tensor_msg(msg).defined());
  EXPECT_FALSE(from_output_tensor_msg(msg).defined());
}

TEST(TorchTensorBridge, DtypeConversionRejectsUnsupportedTriple)
{
  using torch_conversions::detail::DLDataType;
  using torch_conversions::detail::scalar_from_dl_dtype;
  EXPECT_THROW(scalar_from_dl_dtype(DLDataType{kDLFloat, 128, 1}), std::runtime_error);
  EXPECT_THROW(scalar_from_dl_dtype(DLDataType{kDLFloat, 32, 4}), std::runtime_error);
}

TEST(TorchTensorBridge, MakeInputDlpackPopulatesDlTensorFields)
{
  auto msg = allocate_tensor_msg({2, 3}, at::kFloat, c10::kCPU);
  {
    at::Tensor t = torch_conversions::from_output_tensor_msg(*msg);
    t.copy_(torch::arange(0, 6, at::kFloat).reshape({2, 3}));
  }

  auto * dlm = torch_conversions::detail::make_input_dlpack(*msg);
  ASSERT_NE(dlm, nullptr);
  ASSERT_NE(dlm->deleter, nullptr);

  EXPECT_EQ(dlm->dl_tensor.ndim, 2);
  ASSERT_NE(dlm->dl_tensor.shape, nullptr);
  EXPECT_EQ(dlm->dl_tensor.shape[0], 2);
  EXPECT_EQ(dlm->dl_tensor.shape[1], 3);
  ASSERT_NE(dlm->dl_tensor.strides, nullptr);
  EXPECT_EQ(dlm->dl_tensor.strides[0], 3);
  EXPECT_EQ(dlm->dl_tensor.strides[1], 1);

  EXPECT_EQ(dlm->dl_tensor.dtype.code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(dlm->dl_tensor.dtype.bits, 32u);
  EXPECT_EQ(dlm->dl_tensor.dtype.lanes, 1u);

  EXPECT_EQ(static_cast<int32_t>(dlm->dl_tensor.device.device_type),
    static_cast<int32_t>(kDLCPU));
  EXPECT_EQ(dlm->dl_tensor.device.device_id, 0);
  EXPECT_EQ(dlm->dl_tensor.byte_offset, 0u);
  EXPECT_NE(dlm->dl_tensor.data, nullptr);

  dlm->deleter(dlm);
}

TEST(TorchTensorBridge, MakeInputDlpackWithByteOffset)
{
  auto msg = allocate_tensor_msg({16}, at::kInt, c10::kCPU);
  {
    at::Tensor full = torch_conversions::from_output_tensor_msg(*msg);
    for (int i = 0; i < 16; ++i) {
      full.index_put_({i}, i * 100);
    }
  }

  // Capture the base pointer of the allocation, then publish a 4-element
  // view starting at index 4 (16 bytes in).
  auto * base = static_cast<const uint8_t *>(msg->data.data());
  msg->shape = {4};
  msg->strides = {1};
  msg->byte_offset = 4 * sizeof(int32_t);

  auto * dlm = torch_conversions::detail::make_input_dlpack(*msg);
  ASSERT_NE(dlm, nullptr);
  EXPECT_EQ(dlm->dl_tensor.ndim, 1);
  EXPECT_EQ(dlm->dl_tensor.shape[0], 4);

  // The bridge bakes msg->byte_offset into DLTensor::data and sets
  // DLTensor::byte_offset to 0 (portable across DLPack importers that
  // ignore the byte_offset field).
  EXPECT_EQ(dlm->dl_tensor.byte_offset, 0u);
  EXPECT_EQ(static_cast<const uint8_t *>(dlm->dl_tensor.data),
    base + 4 * sizeof(int32_t));

  dlm->deleter(dlm);
}

TEST(TorchTensorBridge, DlpackPtrFreesOnScopeExit)
{
  auto msg = allocate_tensor_msg({3}, at::kInt, c10::kCPU);

  {
    torch_conversions::detail::DlpackPtr holder{
      torch_conversions::detail::make_input_dlpack(*msg)};
    ASSERT_TRUE(holder);
    EXPECT_EQ(holder->dl_tensor.ndim, 1);
    EXPECT_EQ(holder->dl_tensor.shape[0], 3);
  }  // holder destructor invokes deleter; no leak.

  // A second one, this time handed off via release() (simulating a
  // framework's from_dlpack taking ownership).
  torch_conversions::detail::DlpackPtr holder2{
    torch_conversions::detail::make_input_dlpack(*msg)};
  DLManagedTensor * raw = holder2.release();
  ASSERT_NE(raw, nullptr);
  raw->deleter(raw);  // caller takes over ownership
}

TEST(TorchImageBridge, AllocateCpuImageInitializesMetadataAndStorage)
{
  auto msg = allocate_image_msg(2, 3, "rgb8", c10::kCPU);

  EXPECT_EQ(msg->height, 2u);
  EXPECT_EQ(msg->width, 3u);
  EXPECT_EQ(msg->encoding, "rgb8");
  EXPECT_EQ(msg->is_bigendian, 0u);
  EXPECT_EQ(msg->step, 9u);
  EXPECT_EQ(msg->data.size(), 18u);
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");
}

TEST(TorchImageBridge, WritableAndReadOnlyViewsRoundTripAsHwc)
{
  auto msg = allocate_image_msg(2, 3, "rgb8", c10::kCPU);
  {
    at::Tensor output = from_output_image_msg(*msg);
    EXPECT_EQ(output.sizes(), (std::vector<int64_t>{2, 3, 3}));
    EXPECT_EQ(output.strides(), (std::vector<int64_t>{9, 3, 1}));
    EXPECT_EQ(output.scalar_type(), at::kByte);
    output.copy_(torch::arange(0, 18, at::kByte).reshape({2, 3, 3}));
  }

  at::Tensor view = from_input_image_msg(*msg, false);
  EXPECT_EQ(view.data_ptr<uint8_t>(), msg->data.data());
  EXPECT_TRUE(torch::equal(
    view.flatten(), torch::arange(0, 18, at::kByte)));
}

TEST(TorchImageBridge, PaddedRowsProduceStridedHwcView)
{
  ImageMsg msg;
  msg.height = 2;
  msg.width = 2;
  msg.encoding = "rgb8";
  msg.step = 8;
  msg.data.resize(16);

  at::Tensor view = from_output_image_msg(msg);
  EXPECT_EQ(view.sizes(), (std::vector<int64_t>{2, 2, 3}));
  EXPECT_EQ(view.strides(), (std::vector<int64_t>{8, 3, 1}));
  view.fill_(7);
  EXPECT_EQ(msg.data[5], 7u);
  EXPECT_EQ(msg.data[6], 0u);
  EXPECT_EQ(msg.data[7], 0u);
  EXPECT_EQ(msg.data[8], 7u);
}

TEST(TorchImageBridge, CopyPreservesImageMetadataAndHeader)
{
  auto msg = allocate_image_msg(2, 3, "rgb8", c10::kCPU);
  msg->header.frame_id = "camera";
  at::Tensor source = torch::arange(0, 18, at::kByte).reshape({2, 3, 3});

  to_image_msg(*msg, source);

  EXPECT_EQ(msg->header.frame_id, "camera");
  EXPECT_EQ(msg->encoding, "rgb8");
  EXPECT_EQ(msg->step, 9u);
  EXPECT_TRUE(torch::equal(from_input_image_msg(*msg, false), source));
}

TEST(TorchImageBridge, AllocatingCopyDerivesDimensionsFromHwcTensor)
{
  at::Tensor source = torch::ones({4, 5, 3}, at::kByte);
  auto msg = to_image_msg(source, "bgr8");

  EXPECT_EQ(msg->height, 4u);
  EXPECT_EQ(msg->width, 5u);
  EXPECT_EQ(msg->step, 15u);
  EXPECT_EQ(msg->encoding, "bgr8");
  EXPECT_TRUE(torch::equal(from_input_image_msg(*msg, false), source));
}

TEST(TorchImageBridge, SignedByteEncodingProducesCharTensor)
{
  auto msg = allocate_image_msg(2, 3, "8SC1", c10::kCPU);
  EXPECT_EQ(from_output_image_msg(*msg).scalar_type(), at::kChar);
}

TEST(TorchImageBridge, RejectsInvalidImageBoundsAndEncoding)
{
  ImageMsg short_step;
  short_step.height = 2;
  short_step.width = 3;
  short_step.encoding = "rgb8";
  short_step.step = 8;
  short_step.data.resize(16);
  EXPECT_THROW(from_input_image_msg(short_step, false), std::runtime_error);

  ImageMsg short_data;
  short_data.height = 2;
  short_data.width = 3;
  short_data.encoding = "rgb8";
  short_data.step = 9;
  short_data.data.resize(17);
  EXPECT_THROW(from_input_image_msg(short_data, false), std::runtime_error);

  EXPECT_THROW(allocate_image_msg(2, 3, "mono16", c10::kCPU), std::runtime_error);
  EXPECT_THROW(allocate_image_msg(2, 3, "nv12", c10::kCPU), std::runtime_error);
}

TEST(TorchImageBridge, RejectsTensorShapeAndDtypeMismatch)
{
  auto msg = allocate_image_msg(2, 3, "rgb8", c10::kCPU);
  EXPECT_THROW(to_image_msg(*msg, torch::zeros({2, 3}, at::kByte)), std::runtime_error);
  EXPECT_THROW(to_image_msg(*msg, torch::zeros({2, 3, 3}, at::kFloat)), std::runtime_error);
  EXPECT_THROW(to_image_msg(torch::zeros({2, 3}, at::kByte), "rgb8"), std::runtime_error);
}

#ifdef TORCH_CONVERSIONS_HAS_CUDA
TEST(TorchImageBridge, CudaStorageViewsAndAsyncCopyRoundTrip)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime";
  }

  auto stream_guard = torch_conversions::set_stream();
  auto msg = allocate_image_msg(2, 3, "rgb8", c10::kCUDA);
  EXPECT_EQ(msg->data.get_backend_type(), "cuda");

  at::Tensor output = from_output_image_msg(*msg);
  EXPECT_TRUE(output.is_cuda());
  output = {};  // Release the WriteHandle before acquiring subscriber input.

  at::Tensor source = torch::arange(
    0, 18, torch::TensorOptions().dtype(at::kByte).device(c10::kCUDA))
    .reshape({2, 3, 3});
  to_image_msg(*msg, source);

  at::Tensor input = from_input_image_msg(*msg, false);
  EXPECT_TRUE(input.is_cuda());
  EXPECT_EQ(input.sizes(), (std::vector<int64_t>{2, 3, 3}));

  // Synchronization belongs to the materializing test/sink, not conversions.
  torch::cuda::synchronize();
  EXPECT_TRUE(torch::equal(input.cpu(), source.cpu()));
}
#endif

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
