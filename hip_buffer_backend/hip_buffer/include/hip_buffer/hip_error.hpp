// Copyright 2026 Open Source Robotics Foundation, Inc.
// Modified for the HIP port by ObeliskGate, 2026.
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

#ifndef HIP_BUFFER__HIP_ERROR_HPP_
#define HIP_BUFFER__HIP_ERROR_HPP_

#include <hip/hip_runtime_api.h>
#include <rcutils/logging_macros.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace hip_buffer_backend
{

class HipError : public std::runtime_error
{
public:
  explicit HipError(const std::string & message)
  : std::runtime_error(message) {}

  HipError(const char * file, int line, const char * expr, hipError_t err)
  : std::runtime_error(
      format_error(file, line, expr, hipGetErrorName(err), hipGetErrorString(err))) {}

private:
  static std::string format_error(
    const char * file, int line, const char * expr,
    const char * name, const char * desc)
  {
    std::ostringstream oss;
    oss << file << ":" << line << ": " << expr << " failed: "
        << name << " (" << desc << ")";
    return oss.str();
  }
};

inline bool hip_error_is_safe(hipError_t e)
{
  return e == hipSuccess || e == hipErrorDeinitialized;
}

}  // namespace hip_buffer_backend

#define HIP_CHECK(expr) \
  do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
      throw ::hip_buffer_backend::HipError(__FILE__, __LINE__, #expr, _err); \
    } \
  } while (0)

#define HIP_CHECK_NOTHROW(expr, on_error) \
  do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
      RCUTILS_LOG_WARN_NAMED("hip_buffer_backend", \
        "%s:%d: %s failed: %s (%s)", \
        __FILE__, __LINE__, #expr, \
        hipGetErrorName(_err), hipGetErrorString(_err)); \
      (void)hipGetLastError(); \
      on_error; \
    } \
  } while (0)

#endif  // HIP_BUFFER__HIP_ERROR_HPP_
