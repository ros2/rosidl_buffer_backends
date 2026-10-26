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

/// @file intel_buffer_e2e_publisher.cpp
/// @brief Publisher process for test_intel_buffer_e2e: publishes standard
/// sensor_msgs/Image messages whose data is an IntelBuffer, so delivery goes
/// through the RMW / intel_buffer backend path. Run as a separate process so
/// the subscriber exercises cross-process import rather than the backend's
/// intra-process shortcut.
///
/// usage: intel_buffer_e2e_publisher <L0_USM|DMA_BUF|EXTERNAL_MEMMAP|WRAP_DMABUF>
///                                   <topic> <num_frames> <num_subscribers>
///
/// Exit codes: 0 = published all frames, kE2eSkipExitCode = transport not
/// available on this machine, anything else = failure. On success prints
/// "E2E_POOL_TOTAL_BLOCKS=<n>" (pool transports) to stdout.

#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "intel_buffer/intel_buffer_api.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "intel_buffer_e2e_common.hpp"

using intel_buffer_backend::IntelBufferImpl;
using intel_buffer_backend::TransportMode;

namespace
{

int skip(const std::string & reason)
{
  std::printf("E2E_SKIP: %s\n", reason.c_str());
  std::fflush(stdout);
  return e2e::kE2eSkipExitCode;
}

bool has_transport(rosidl::Buffer<uint8_t> & buffer, TransportMode want)
{
  auto * impl = dynamic_cast<IntelBufferImpl<uint8_t> *>(buffer.get_impl());
  return impl && impl->get_block() && impl->get_block()->ptr &&
         impl->get_block()->transport == want;
}

// dma-heap buffers standing in for V4L2 capture buffers (camera path).
std::vector<int> alloc_dma_heap_buffers(size_t count, size_t bytes)
{
  std::vector<int> fds;
  int heap = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
  if (heap < 0) {
    return fds;
  }
  for (size_t i = 0; i < count; ++i) {
    struct dma_heap_allocation_data alloc{};
    alloc.len = bytes;
    alloc.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
      break;
    }
    fds.push_back(static_cast<int>(alloc.fd));
  }
  close(heap);
  return fds;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 5) {
    std::fprintf(stderr,
      "usage: %s <L0_USM|DMA_BUF|EXTERNAL_MEMMAP|WRAP_DMABUF> <topic> "
      "<num_frames> <num_subscribers>\n", argv[0]);
    return 2;
  }
  const std::string transport = argv[1];
  const std::string topic = argv[2];
  const uint32_t num_frames = static_cast<uint32_t>(std::stoul(argv[3]));
  const size_t num_subscribers = std::stoul(argv[4]);
  const size_t frame_bytes = e2e::kStep * e2e::kHeight;

  // The pool reads its transport once, when the backend first creates it.
  if (transport == "DMA_BUF") {
    setenv("INTEL_MEMORY_BUFFER_TYPE", "dmabuf", 1);
  } else if (transport == "L0_USM") {
    setenv("INTEL_MEMORY_BUFFER_TYPE", "l0usm", 1);
  } else if (transport != "EXTERNAL_MEMMAP" && transport != "WRAP_DMABUF") {
    std::fprintf(stderr, "unknown transport '%s'\n", transport.c_str());
    return 2;
  }

  // Source buffers for the transports that do not allocate from the pool.
  intel_buffer_backend::ExternalMemMapRing ring;
  std::vector<int> dma_fds;
  if (transport == "EXTERNAL_MEMMAP" &&
    !ring.create("/ibb_e2e_" + std::to_string(getpid()), frame_bytes))
  {
    return skip("Level Zero external-memmap extension unavailable");
  }
  if (transport == "WRAP_DMABUF") {
    dma_fds = alloc_dma_heap_buffers(e2e::kWrapDmabufSlots, frame_bytes);
    if (dma_fds.size() != e2e::kWrapDmabufSlots) {
      return skip("cannot allocate from /dev/dma_heap/system");
    }
  }

  // One frame's Image with data backed by an IntelBuffer for `transport`.
  auto make_buffer = [&](uint32_t frame) -> rosidl::Buffer<uint8_t> {
      if (transport == "EXTERNAL_MEMMAP") {
        const size_t slot = ring.advance();
        e2e::fill_pattern(ring.data(slot), frame_bytes, frame);
        return ring.wrap(slot, frame_bytes);
      }
      if (transport == "WRAP_DMABUF") {
        const size_t slot = frame % dma_fds.size();
        auto buffer = intel_buffer_backend::wrap_dmabuf(dma_fds[slot], frame_bytes, false);
        auto * impl = dynamic_cast<IntelBufferImpl<uint8_t> *>(buffer.get_impl());
        if (impl && impl->get_block()) {
          // Same as the camera node: key the fd export on the capture slot.
          impl->get_block()->block_id = static_cast<uint32_t>(slot);
        }
        e2e::fill_pattern(intel_buffer_backend::get_usm_ptr(buffer), frame_bytes, frame);
        return buffer;
      }
      auto buffer = intel_buffer_backend::allocate_buffer(frame_bytes);
      e2e::fill_pattern(intel_buffer_backend::get_usm_ptr(buffer), frame_bytes, frame);
      return buffer;
    };

  rclcpp::init(argc, argv);
  int rc = 0;
  {
    auto node = std::make_shared<rclcpp::Node>("intel_buffer_e2e_publisher");
    auto pub = node->create_publisher<sensor_msgs::msg::Image>(topic, e2e::qos());

    // The pool exists once the publisher endpoint is created; make sure it
    // actually provides the requested transport before testing it.
    if (transport == "L0_USM" || transport == "DMA_BUF") {
      auto probe = intel_buffer_backend::allocate_buffer(frame_bytes);
      const TransportMode want = transport == "L0_USM" ?
        TransportMode::LEVEL_ZERO_USM : TransportMode::DMA_BUF;
      if (!has_transport(probe, want)) {
        rc = skip("buffer pool cannot provide " + transport);
      }
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (rc == 0 && pub->get_subscription_count() < num_subscribers) {
      if (std::chrono::steady_clock::now() > deadline) {
        std::fprintf(stderr, "timed out waiting for %zu subscribers\n", num_subscribers);
        rc = 3;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    // Let backend negotiation with the new endpoints settle before frame 0.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    for (uint32_t frame = 0; rc == 0 && frame < num_frames; ++frame) {
      sensor_msgs::msg::Image msg;
      msg.header.frame_id = std::to_string(frame);
      msg.height = e2e::kHeight;
      msg.width = e2e::kWidth;
      msg.encoding = e2e::kEncoding;
      msg.step = e2e::kStep;
      msg.data = make_buffer(frame);
      if (intel_buffer_backend::get_usm_ptr(msg.data) == nullptr) {
        std::fprintf(stderr, "frame %u: buffer allocation failed\n", frame);
        rc = 4;
        break;
      }
      pub->publish(msg);
      std::this_thread::sleep_for(e2e::kFramePeriod);
    }

    if (rc == 0) {
      // Give subscribers time to take and release the last frames.
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (transport == "L0_USM" || transport == "DMA_BUF") {
        auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
        std::printf("E2E_POOL_TOTAL_BLOCKS=%zu\n", pool ? pool->total_blocks() : 0);
        std::fflush(stdout);
      }
    }
  }
  rclcpp::shutdown();
  for (int fd : dma_fds) {
    close(fd);
  }
  return rc;
}
