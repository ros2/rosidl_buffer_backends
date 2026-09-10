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

// This translation unit is compiled only against a CUDA LibTorch, so it can
// reach for a real stream the way a GPU consumer of the adapter would.
#include <c10/cuda/CUDAStream.h>

#include "torch_conversions/torch_conversions.hpp"

namespace
{

void * current_stream()
{
  return c10::cuda::getCurrentCUDAStream().stream();
}

}  // namespace

TEST(TorchConversionsCuda, AllocatesCudaStorage)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }

  auto msg = torch_conversions::allocate_tensor_msg(
    {4}, at::kFloat, c10::kCUDA);
  EXPECT_EQ(msg->data.get_backend_type(), "cuda");

  auto tensor = torch_conversions::from_output_tensor_msg(*msg);
  EXPECT_TRUE(tensor.is_cuda());
  tensor.fill_(1.0);
}

TEST(TorchConversionsCuda, RoundTripsOnACallerSuppliedStream)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }

  auto msg = torch_conversions::allocate_tensor_msg(
    {64}, at::kFloat, c10::kCUDA);

  {
    auto output = torch_conversions::from_output_tensor_msg(
      *msg, current_stream());
    ASSERT_TRUE(output.is_cuda());
    output.fill_(3.0);
  }

  auto input = torch_conversions::from_input_tensor_msg(
    *msg, /*clone=*/true, current_stream());
  ASSERT_TRUE(input.is_cuda());
  EXPECT_FLOAT_EQ(input.sum().item<float>(), 192.0f);
}

TEST(TorchConversionsCuda, CopiesIntoStorageOnACallerSuppliedStream)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }

  const auto source = at::full({32}, 5.0, at::device(at::kCUDA).dtype(at::kFloat));

  auto msg = torch_conversions::to_tensor_msg(source, current_stream());
  ASSERT_EQ(msg->data.get_backend_type(), "cuda");

  auto readback = torch_conversions::from_input_tensor_msg(
    *msg, /*clone=*/true, current_stream());
  EXPECT_FLOAT_EQ(readback.sum().item<float>(), 160.0f);
}

// A caller running off the default stream is the case the parameter exists
// for: the storage plugin has to order against the stream that was passed,
// not against whatever stream happens to be current inside the adapter.
TEST(TorchConversionsCuda, HonoursANonDefaultStream)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }

  const auto stream = c10::cuda::getStreamFromPool();
  c10::cuda::CUDAStreamGuard guard(stream);
  ASSERT_NE(stream.stream(), c10::cuda::getDefaultCUDAStream().stream());

  auto msg = torch_conversions::allocate_tensor_msg(
    {32}, at::kFloat, c10::kCUDA);

  {
    auto output = torch_conversions::from_output_tensor_msg(
      *msg, stream.stream());
    output.fill_(7.0);
  }

  auto input = torch_conversions::from_input_tensor_msg(
    *msg, /*clone=*/true, stream.stream());
  EXPECT_FLOAT_EQ(input.sum().item<float>(), 224.0f);
}
