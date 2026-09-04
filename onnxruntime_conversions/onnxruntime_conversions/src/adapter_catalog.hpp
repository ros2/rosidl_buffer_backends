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

#ifndef ADAPTER_CATALOG_HPP_
#define ADAPTER_CATALOG_HPP_

#include <map>
#include <stdexcept>
#include <string>

namespace onnxruntime_conversions::detail
{

inline void register_adapter_id(
  std::map<std::string, std::string> & adapter_classes,
  const std::string & adapter_id,
  const std::string & plugin_class)
{
  const auto [existing, inserted] =
    adapter_classes.emplace(adapter_id, plugin_class);
  if (!inserted) {
    throw std::runtime_error(
            "Duplicate ONNX Runtime conversion adapter ID '" + adapter_id +
            "' returned by plugin classes '" + existing->second + "' and '" +
            plugin_class + "'");
  }
}

}  // namespace onnxruntime_conversions::detail

#endif  // ADAPTER_CATALOG_HPP_
