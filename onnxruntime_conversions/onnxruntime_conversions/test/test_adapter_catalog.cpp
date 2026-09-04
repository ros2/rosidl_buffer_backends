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

#include <map>
#include <string>

#include "adapter_catalog.hpp"

TEST(OnnxRuntimeConversionsCore, RejectsDuplicateAdapterIds)
{
  std::map<std::string, std::string> adapter_classes;
  onnxruntime_conversions::detail::register_adapter_id(
    adapter_classes, "duplicate", "plugins/First");
  try {
    onnxruntime_conversions::detail::register_adapter_id(
      adapter_classes, "duplicate", "plugins/Second");
    FAIL() << "Duplicate adapter ID did not throw";
  } catch (const std::runtime_error & error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("Duplicate ONNX Runtime conversion adapter ID 'duplicate'"),
      std::string::npos);
    EXPECT_NE(message.find("'plugins/First'"), std::string::npos);
    EXPECT_NE(message.find("'plugins/Second'"), std::string::npos);
  }
}
