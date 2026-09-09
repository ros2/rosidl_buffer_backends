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

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "torch_conversions/detail/torch_conversions_impl.hpp"

namespace torch_conversions
{
namespace detail
{

uintptr_t current_stream()
{
  return reinterpret_cast<uintptr_t>(
    at::cuda::getCurrentCUDAStream().stream());
}

}  // namespace detail

class StreamGuard::Impl
{
public:
  Impl()
  : guard_(torch::cuda::is_available() ?
      std::optional<c10::Stream>(c10::cuda::getStreamFromPool()) :
      std::nullopt) {}

private:
  c10::OptionalStreamGuard guard_;
};

StreamGuard::StreamGuard()
: impl_(std::make_unique<Impl>()) {}

StreamGuard::~StreamGuard() = default;
StreamGuard::StreamGuard(StreamGuard &&) noexcept = default;
StreamGuard & StreamGuard::operator=(StreamGuard &&) noexcept = default;

}  // namespace torch_conversions
