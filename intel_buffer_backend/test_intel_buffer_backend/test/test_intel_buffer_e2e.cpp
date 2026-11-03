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

/// @file test_intel_buffer_e2e.cpp
/// @brief End-to-end rosidl::Buffer integration test: a separate publisher
/// process (intel_buffer_e2e_publisher) publishes standard sensor_msgs/Image
/// messages with IntelBuffer-backed data; this process subscribes through RMW.
/// Nothing calls the backend directly — descriptor creation and import happen
/// only via the RMW/backend callbacks, as in a real application.
///
/// Per transport it checks that
///  - a subscriber accepting "intel_buffer" receives data on that backend, via
///    the requested transport, with intact content and Image metadata;
///  - a plain subscriber (CPU only, the default) on the same topic receives the
///    same bytes, converted to CPU automatically;
///  - consumer references are released (the publisher's pool does not grow).
///
/// A separate case holds each EXTERNAL_MEMMAP frame longer than a full ring
/// cycle, to check the producer never overwrites a slot a consumer still uses.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "intel_buffer/intel_buffer_api.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "intel_buffer_e2e_common.hpp"

using intel_buffer_backend::IntelBufferImpl;
using intel_buffer_backend::TransportMode;
using sensor_msgs::msg::Image;

namespace
{

struct Received
{
  std::set<uint32_t> frames;
  std::vector<std::string> errors;

  void error(uint32_t frame, const std::string & what)
  {
    errors.push_back("frame " + std::to_string(frame) + ": " + what);
  }
};

// Common checks; returns the frame index carried in header.frame_id.
uint32_t check_image(const Image & msg, Received & rx)
{
  uint32_t frame = 0;
  try {
    frame = static_cast<uint32_t>(std::stoul(msg.header.frame_id));
  } catch (const std::exception &) {
    rx.errors.push_back("bad frame_id '" + msg.header.frame_id + "'");
    return frame;
  }
  rx.frames.insert(frame);
  if (msg.width != e2e::kWidth || msg.height != e2e::kHeight || msg.step != e2e::kStep ||
    msg.encoding != e2e::kEncoding)
  {
    rx.error(frame, "Image metadata not preserved (" + std::to_string(msg.width) + "x" +
      std::to_string(msg.height) + " step " + std::to_string(msg.step) + " " + msg.encoding + ")");
  }
  if (msg.data.size() != static_cast<size_t>(e2e::kStep) * e2e::kHeight) {
    rx.error(frame, "data size " + std::to_string(msg.data.size()));
  }
  return frame;
}

void check_content(const uint8_t * data, size_t n, uint32_t frame, Received & rx)
{
  if (data == nullptr) {
    rx.error(frame, "no data pointer");
    return;
  }
  size_t bad = e2e::first_mismatch(data, n, frame);
  if (bad != n) {
    rx.error(frame, "content mismatch at byte " + std::to_string(bad));
  }
}

TransportMode expected_transport(const std::string & transport)
{
  if (transport == "L0_USM") {
    return TransportMode::LEVEL_ZERO_USM;
  }
  if (transport == "EXTERNAL_MEMMAP") {
    return TransportMode::EXTERNAL_MEMMAP;
  }
  return TransportMode::DMA_BUF;  // DMA_BUF, WRAP_DMABUF
}

std::string describe(const std::vector<std::string> & errors)
{
  std::string out;
  for (size_t i = 0; i < errors.size() && i < 5; ++i) {
    out += "\n  " + errors[i];
  }
  if (errors.size() > 5) {
    out += "\n  ... (" + std::to_string(errors.size() - 5) + " more)";
  }
  return out;
}

/// The publisher process, with its stdout captured.
class PublisherProcess
{
public:
  bool start(
    const std::string & transport, const std::string & topic, size_t num_subscribers = 2,
    std::chrono::milliseconds frame_period = e2e::kFramePeriod)
  {
    // Build argv before fork(): only exec-safe calls in the child.
    std::vector<std::string> args = {
      INTEL_BUFFER_E2E_PUBLISHER, transport, topic, std::to_string(e2e::kNumFrames),
      std::to_string(num_subscribers), std::to_string(frame_period.count())};
    std::vector<char *> argv;
    for (auto & a : args) {
      argv.push_back(a.data());
    }
    argv.push_back(nullptr);

    int fds[2];
    if (pipe(fds) != 0) {
      return false;
    }
    pid_ = fork();
    if (pid_ == 0) {
      dup2(fds[1], STDOUT_FILENO);
      close(fds[0]);
      close(fds[1]);
      execv(argv[0], argv.data());
      _exit(127);
    }
    close(fds[1]);
    out_fd_ = fds[0];
    return pid_ > 0;
  }

  /// Non-blocking; true once the process has exited (status in exit_code()).
  bool exited()
  {
    if (pid_ <= 0) {
      return true;
    }
    int status = 0;
    if (waitpid(pid_, &status, WNOHANG) != pid_) {
      return false;
    }
    exit_code_ = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    pid_ = -1;
    return true;
  }

  void kill_now()
  {
    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      waitpid(pid_, nullptr, 0);
      pid_ = -1;
    }
  }

  std::string output()
  {
    std::string out;
    char buf[512];
    ssize_t n;
    while (out_fd_ >= 0 && (n = read(out_fd_, buf, sizeof(buf))) > 0) {
      out.append(buf, static_cast<size_t>(n));
    }
    return out;
  }

  int exit_code() const {return exit_code_;}

  ~PublisherProcess()
  {
    kill_now();
    if (out_fd_ >= 0) {
      close(out_fd_);
    }
  }

private:
  pid_t pid_{-1};
  int out_fd_{-1};
  int exit_code_{-1};
};

}  // namespace

class IntelBufferE2E : public ::testing::TestWithParam<std::string>
{
protected:
  void SetUp() override {rclcpp::init(0, nullptr);}
  void TearDown() override {rclcpp::shutdown();}
};

TEST_P(IntelBufferE2E, StandardImageRoundTripThroughRmw)
{
  const std::string transport = GetParam();
  const std::string topic = "/intel_buffer_e2e/" + transport;
  const size_t frame_bytes = static_cast<size_t>(e2e::kStep) * e2e::kHeight;
  const TransportMode want = expected_transport(transport);

  auto node = std::make_shared<rclcpp::Node>("intel_buffer_e2e_subscriber");
  Received intel_rx;
  Received cpu_rx;

  rclcpp::SubscriptionOptions intel_opts;
  intel_opts.acceptable_buffer_backends = "intel_buffer";
  auto intel_sub = node->create_subscription<Image>(
    topic, e2e::qos(),
    [&](const Image::SharedPtr msg) {
      uint32_t frame = check_image(*msg, intel_rx);
      if (msg->data.get_backend_type() != "intel_buffer") {
        intel_rx.error(frame, "delivered on backend '" + msg->data.get_backend_type() + "'");
        return;
      }
      const auto * impl =
      dynamic_cast<const IntelBufferImpl<uint8_t> *>(msg->data.get_impl());
      if (!impl || !impl->get_block() || impl->get_block()->transport != want) {
        intel_rx.error(frame, "not imported via the " + transport + " transport");
      }
      check_content(intel_buffer_backend::get_usm_ptr(msg->data), frame_bytes, frame, intel_rx);
    }, intel_opts);

  // Default options: CPU only — the backend must convert for this endpoint.
  auto cpu_sub = node->create_subscription<Image>(
    topic, e2e::qos(),
    [&](const Image::SharedPtr msg) {
      uint32_t frame = check_image(*msg, cpu_rx);
      if (msg->data.get_backend_type() != "cpu") {
        cpu_rx.error(frame, "delivered on backend '" + msg->data.get_backend_type() + "'");
        return;
      }
      auto bytes = msg->data.to_vector();
      check_content(bytes.data(), bytes.size(), frame, cpu_rx);
    });

  PublisherProcess publisher;
  ASSERT_TRUE(publisher.start(transport, topic));

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  bool timed_out = false;
  while (!publisher.exited()) {
    if (std::chrono::steady_clock::now() > deadline) {
      publisher.kill_now();
      timed_out = true;
      break;
    }
    exec.spin_some(std::chrono::milliseconds(20));
  }
  // Drain anything still in flight.
  for (int i = 0; i < 10; ++i) {
    exec.spin_some(std::chrono::milliseconds(20));
  }

  const std::string out = publisher.output();
  if (!timed_out && publisher.exit_code() == e2e::kE2eSkipExitCode) {
    GTEST_SKIP() << transport << " not available on this machine: " << out;
  }

  EXPECT_TRUE(intel_rx.errors.empty()) << "intel_buffer subscriber:" << describe(intel_rx.errors);
  EXPECT_TRUE(cpu_rx.errors.empty()) << "CPU subscriber:" << describe(cpu_rx.errors);
  EXPECT_EQ(intel_rx.frames.size(), e2e::kNumFrames) << "intel_buffer subscriber missed frames";
  EXPECT_EQ(cpu_rx.frames.size(), e2e::kNumFrames) << "CPU subscriber missed frames";

  // Pool transports: a consumer that never releases its reference keeps every
  // block busy, so the publisher's pool grows by one block per frame instead
  // of recycling a handful (and its teardown then waits on those references).
  const std::string key = "E2E_POOL_TOTAL_BLOCKS=";
  const auto pos = out.find(key);
  const bool published_all = pos != std::string::npos;
  if (published_all) {
    const size_t total_blocks = std::stoul(out.substr(pos + key.size()));
    EXPECT_LE(total_blocks, e2e::kNumFrames / 2)
      << "publisher pool grew to " << total_blocks << " blocks for " << e2e::kNumFrames
      << " frames: subscriber references are not being released";
  }

  if (timed_out) {
    FAIL() << (published_all ?
      "publisher published every frame but hung on exit: its buffer pool waits for "
      "subscriber references that were never released" :
      "publisher process did not finish within 60 s") << "; stdout:\n" << out;
  }
  ASSERT_EQ(publisher.exit_code(), 0) << "publisher failed; stdout:\n" << out;
  if (transport == "L0_USM" || transport == "DMA_BUF") {
    EXPECT_TRUE(published_all) << "publisher did not report pool size; stdout:\n" << out;
  }
}

INSTANTIATE_TEST_SUITE_P(
  Transports, IntelBufferE2E,
  ::testing::Values("L0_USM", "DMA_BUF", "EXTERNAL_MEMMAP", "WRAP_DMABUF"),
  [](const ::testing::TestParamInfo<std::string> & info) {return info.param;});

class IntelBufferE2ESlowConsumer : public ::testing::Test
{
protected:
  void SetUp() override {rclcpp::init(0, nullptr);}
  void TearDown() override {rclcpp::shutdown();}
};

// The EXTERNAL_MEMMAP ring has 4 slots; at a 10 ms frame period each slot comes
// round again every 40 ms. This subscriber holds every frame for 100 ms (like
// a slow inference node) and re-checks its content before releasing it. The
// producer must not reuse a slot a consumer holds. Frames that went stale while
// queued may be rejected (delivered empty), but a delivered frame must never
// change under its reader.
TEST_F(IntelBufferE2ESlowConsumer, ExternalMemMapSlotNotOverwrittenWhileHeld)
{
  const std::string topic = "/intel_buffer_e2e/slow_consumer";
  const size_t frame_bytes = static_cast<size_t>(e2e::kStep) * e2e::kHeight;
  constexpr auto kHold = std::chrono::milliseconds(100);

  auto node = std::make_shared<rclcpp::Node>("intel_buffer_e2e_slow_subscriber");
  Received rx;
  size_t intact = 0;
  size_t stale = 0;

  rclcpp::SubscriptionOptions opts;
  opts.acceptable_buffer_backends = "intel_buffer";
  auto sub = node->create_subscription<Image>(
    topic, e2e::qos(),
    [&](const Image::SharedPtr msg) {
      if (msg->data.size() == 0) {
        ++stale;  // slot reused before this queued frame was taken; rejected
        return;
      }
      const size_t errors_before = rx.errors.size();
      uint32_t frame = check_image(*msg, rx);
      const auto * impl =
      dynamic_cast<const IntelBufferImpl<uint8_t> *>(msg->data.get_impl());
      if (!impl || !impl->get_block() ||
        impl->get_block()->transport != TransportMode::EXTERNAL_MEMMAP)
      {
        rx.error(frame, "not imported via the EXTERNAL_MEMMAP transport");
        return;
      }
      const uint8_t * data = intel_buffer_backend::get_usm_ptr(msg->data);
      check_content(data, frame_bytes, frame, rx);
      std::this_thread::sleep_for(kHold);
      const size_t bad = data ? e2e::first_mismatch(data, frame_bytes, frame) : 0;
      if (bad != frame_bytes) {
        rx.error(frame, "overwritten while held (byte " + std::to_string(bad) + ")");
      }
      if (rx.errors.size() == errors_before) {
        ++intact;
      }
    }, opts);

  PublisherProcess publisher;
  ASSERT_TRUE(publisher.start("EXTERNAL_MEMMAP", topic, 1, std::chrono::milliseconds(10)));

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (!publisher.exited() && std::chrono::steady_clock::now() < deadline) {
    exec.spin_some(std::chrono::milliseconds(20));
  }
  publisher.kill_now();

  const std::string out = publisher.output();
  if (publisher.exit_code() == e2e::kE2eSkipExitCode) {
    GTEST_SKIP() << "EXTERNAL_MEMMAP not available on this machine: " << out;
  }
  ASSERT_EQ(publisher.exit_code(), 0) << "publisher failed; stdout:\n" << out;
  EXPECT_TRUE(rx.errors.empty()) << "slow subscriber:" << describe(rx.errors);
  EXPECT_GE(intact, 2u) << "too few frames delivered (" << stale << " rejected as stale)";
}
