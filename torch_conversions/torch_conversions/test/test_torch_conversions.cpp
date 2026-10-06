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

#include <stdexcept>

#include "torch_conversions/torch_conversions.hpp"

TEST(TorchConversions, EmptyDataReturnsUndefinedTensor)
{
  torch_conversions::TensorMsg msg;
  EXPECT_FALSE(torch_conversions::from_input_tensor_msg(msg).defined());
  EXPECT_FALSE(torch_conversions::from_output_tensor_msg(msg).defined());
}

TEST(TorchConversions, RejectsDeviceWithoutConversionPlugin)
{
  if (torch_conversions::backend_available("cuda")) {
    GTEST_SKIP() << "the CUDA conversion plugin is installed";
  }
  EXPECT_THROW(
    torch_conversions::set_stream(c10::kCUDA), std::runtime_error);
  EXPECT_THROW(
    torch_conversions::allocate_tensor_msg(
      {1}, at::kFloat, c10::kCUDA),
    std::runtime_error);
}
