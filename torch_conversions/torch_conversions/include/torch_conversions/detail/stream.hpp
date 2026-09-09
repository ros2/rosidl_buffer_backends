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

#ifndef TORCH_CONVERSIONS__DETAIL__STREAM_HPP_
#define TORCH_CONVERSIONS__DETAIL__STREAM_HPP_

// TORCH_CONVERSIONS_ENABLE_CUDA is set by the package config when the LibTorch
// the consumer builds against provides torch_cuda. A CPU-only LibTorch still
// ships the ATen CUDA headers, so the target must be probed, not the include.
#ifdef TORCH_CONVERSIONS_ENABLE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace torch_conversions
{
namespace detail
{

/// Raw handle of the accelerator stream LibTorch is currently using, or 0 for
/// an accelerator-free build.
inline uintptr_t current_stream()
{
#ifdef TORCH_CONVERSIONS_ENABLE_CUDA
  return reinterpret_cast<uintptr_t>(at::cuda::getCurrentCUDAStream().stream());
#else
  return 0;
#endif
}

}  // namespace detail

/// Runs the enclosing scope on a pooled accelerator stream.
class StreamGuard
{
public:
  StreamGuard();
  // Declared here and defaulted out of line so the guard stays non-trivial and
  // callers that only hold it for its scope do not draw unused-variable
  // warnings.
  ~StreamGuard();

  StreamGuard(StreamGuard &&) noexcept = default;
  StreamGuard & operator=(StreamGuard &&) noexcept = default;
  StreamGuard(const StreamGuard &) = delete;
  StreamGuard & operator=(const StreamGuard &) = delete;

private:
#ifdef TORCH_CONVERSIONS_ENABLE_CUDA
  std::unique_ptr<c10::OptionalStreamGuard> guard_;
#endif
};

inline StreamGuard::StreamGuard()
#ifdef TORCH_CONVERSIONS_ENABLE_CUDA
: guard_(std::make_unique<c10::OptionalStreamGuard>(
      torch::cuda::is_available() ?
      std::optional<c10::Stream>(c10::cuda::getStreamFromPool()) :
      std::nullopt))
#endif
{
}

inline StreamGuard::~StreamGuard() = default;

inline StreamGuard set_stream()
{
  return StreamGuard();
}

}  // namespace torch_conversions

#endif  // TORCH_CONVERSIONS__DETAIL__STREAM_HPP_
