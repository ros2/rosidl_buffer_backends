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

#ifndef TEST_INTEL_MEMORY_BUFFER_BACKEND__LATENCY_STATS_HPP_
#define TEST_INTEL_MEMORY_BUFFER_BACKEND__LATENCY_STATS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

struct LatencyStats
{
  explicit LatencyStats(std::size_t warmup = 0) : warmup_(warmup) {}

  std::vector<double> samples;

  void add(double ms)
  {
    if (skipped_ < warmup_) {
      ++skipped_;
      return;
    }
    samples.push_back(ms);
  }
  std::size_t count() const { return samples.size(); }

  void report(const rclcpp::Logger & logger, const char * name)
  {
    if (samples.empty()) {
      return;
    }
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();

    const double mn = samples.front();
    const double mx = samples.back();
    double sum = 0.0;
    for (double v : samples) {
      sum += v;
    }
    const double avg = sum / static_cast<double>(n);
    double var = 0.0;
    for (double v : samples) {
      var += (v - avg) * (v - avg);
    }
    const double stddev = std::sqrt(var / static_cast<double>(n));

    auto pct = [&](double p) {
        std::size_t idx =
          static_cast<std::size_t>(std::ceil(p / 100.0 * static_cast<double>(n)));
        if (idx > 0) {
          --idx;
        }
        if (idx >= n) {
          idx = n - 1;
        }
        return samples[idx];
      };

    char buf[1024];
    std::snprintf(buf, sizeof(buf),
      "%s (over %zu frames)\n"
      "+-----------+-----------+-----------+-----------+-----------+"
      "-----------+-----------+-----------+\n"
      "| %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s |\n"
      "+-----------+-----------+-----------+-----------+-----------+"
      "-----------+-----------+-----------+\n"
      "| %9.3f | %9.3f | %9.3f | %9.3f | %9.3f | %9.3f | %9.3f | %9.3f |\n"
      "+-----------+-----------+-----------+-----------+-----------+"
      "-----------+-----------+-----------+",
      name, n,
      "min", "avg", "stddev", "max", "spread", "P50", "P95", "P99",
      mn, avg, stddev, mx, mx - mn, pct(50.0), pct(95.0), pct(99.0));
    RCLCPP_INFO(logger, "%s", buf);

    samples.clear();
  }

private:
  std::size_t warmup_{0};   // number of initial samples to drop (startup transient)
  std::size_t skipped_{0};  // how many have been dropped so far
};

#endif  // TEST_INTEL_MEMORY_BUFFER_BACKEND__LATENCY_STATS_HPP_
