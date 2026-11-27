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

#ifndef INTEL_MEMORY_CORE__LOGGING_HPP_
#define INTEL_MEMORY_CORE__LOGGING_HPP_

#include <cstdio>

#define INTEL_MEMORY_LOG(level, name, ...)         \
  do {                                             \
    std::fprintf(stderr, "[%s] [%s] ", level, name); \
    std::fprintf(stderr, __VA_ARGS__);             \
    std::fprintf(stderr, "\n");                    \
  } while (0)

#define RCUTILS_LOG_DEBUG_NAMED(name, ...) INTEL_MEMORY_LOG("DEBUG", name, __VA_ARGS__)
#define RCUTILS_LOG_INFO_NAMED(name, ...)  INTEL_MEMORY_LOG("INFO", name, __VA_ARGS__)
#define RCUTILS_LOG_WARN_NAMED(name, ...)  INTEL_MEMORY_LOG("WARN", name, __VA_ARGS__)
#define RCUTILS_LOG_ERROR_NAMED(name, ...) INTEL_MEMORY_LOG("ERROR", name, __VA_ARGS__)

#endif  // INTEL_MEMORY_CORE__LOGGING_HPP_
