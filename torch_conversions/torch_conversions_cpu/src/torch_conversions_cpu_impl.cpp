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

#include <cstdint>
#include <memory>

#include "torch_conversions/detail/torch_conversions_impl.hpp"

namespace torch_conversions
{
namespace detail
{

uintptr_t current_stream()
{
  return 0;
}

}  // namespace detail

class StreamGuard::Impl {};

StreamGuard::StreamGuard()
: impl_(std::make_unique<Impl>()) {}

StreamGuard::~StreamGuard() = default;
StreamGuard::StreamGuard(StreamGuard &&) noexcept = default;
StreamGuard & StreamGuard::operator=(StreamGuard &&) noexcept = default;

}  // namespace torch_conversions
