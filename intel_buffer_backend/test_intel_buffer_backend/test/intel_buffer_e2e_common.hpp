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

/// @file intel_buffer_e2e_common.hpp
/// @brief Frame layout and content pattern shared by the end-to-end test's
/// publisher process and subscriber test.

#ifndef INTEL_BUFFER_E2E_COMMON_HPP_
#define INTEL_BUFFER_E2E_COMMON_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>

#include "rclcpp/qos.hpp"

namespace e2e
{

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;
constexpr const char * kEncoding = "bgr8";
constexpr uint32_t kStep = kWidth * 3;
constexpr uint32_t kNumFrames = 40;
// Slow enough that EXTERNAL_MEMMAP / WRAP_DMABUF slots (reused every few
// frames, like a camera ring) are not overwritten while a subscriber reads.
constexpr std::chrono::milliseconds kFramePeriod{50};
constexpr size_t kWrapDmabufSlots = 4;
constexpr int kE2eSkipExitCode = 77;

// Distinct per byte and per frame, so stale or mixed-up frames are detected.
inline uint8_t pattern_byte(size_t i, uint32_t frame)
{
  return static_cast<uint8_t>(i * 7 + frame * 13 + 1);
}

inline void fill_pattern(uint8_t * data, size_t n, uint32_t frame)
{
  for (size_t i = 0; i < n; ++i) {
    data[i] = pattern_byte(i, frame);
  }
}

/// Index of the first byte not matching the pattern, or n if all match.
inline size_t first_mismatch(const uint8_t * data, size_t n, uint32_t frame)
{
  for (size_t i = 0; i < n; ++i) {
    if (data[i] != pattern_byte(i, frame)) {
      return i;
    }
  }
  return n;
}

inline rclcpp::QoS qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(kNumFrames)).reliable();
}

}  // namespace e2e

#endif  // INTEL_BUFFER_E2E_COMMON_HPP_
