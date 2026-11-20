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

#include <stdexcept>

#include "executorch_conversions/executorch_conversions.hpp"

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;

TEST(ExecuTorchConversions, EmptyDataReturnsNullTensor)
{
  executorch_conversions::TensorMsg msg;
  EXPECT_EQ(executorch_conversions::from_input_tensor_msg(msg), nullptr);
  EXPECT_EQ(executorch_conversions::from_output_tensor_msg(msg), nullptr);
}

TEST(ExecuTorchConversions, RejectsDeviceWithoutConversionPlugin)
{
  if (executorch_conversions::backend_available("cuda")) {
    GTEST_SKIP() << "the CUDA conversion plugin is installed";
  }
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg(
      {1}, ScalarType::Float, Device(DeviceType::CUDA)),
    std::runtime_error);
}

TEST(ExecuTorchConversions, RejectsUnsupportedDtype)
{
  executorch_conversions::TensorMsg msg;
  msg.dtype_code = 5;  // DLPack complex
  msg.dtype_bits = 64;
  msg.dtype_lanes = 1;
  EXPECT_THROW(executorch_conversions::scalar_type(msg), std::runtime_error);
  msg.dtype_code = 2;
  msg.dtype_bits = 32;
  msg.dtype_lanes = 4;
  EXPECT_THROW(executorch_conversions::scalar_type(msg), std::runtime_error);
}
