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

#include <stdexcept>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

namespace
{

using onnxruntime_conversions::allocate_tensor_msg;
using onnxruntime_conversions::backend_available;

TEST(OnnxRuntimeConversions, UnavailableBackendThrows)
{
  if (backend_available("cuda")) {
    GTEST_SKIP() << "the CUDA conversion plugin is installed";
  }
  EXPECT_THROW(
    allocate_tensor_msg({1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"),
    std::runtime_error);
}

}  // namespace
