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

#include "cuda_buffer_backend/ipc_metadata.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "rcutils/logging_macros.h"

namespace cuda_buffer_backend
{
namespace detail
{
namespace
{
constexpr const char * kVersion = "cuda-ipc-v1";
constexpr size_t kMaxMetadataSize = 1024;

bool hex_id(const std::string & value)
{
  return value.size() == 32 && std::all_of(value.begin(), value.end(), [](char c) {
             return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

bool valid(const IpcMetadata & value)
{
  return hex_id(value.process) && hex_id(value.gpu) &&
         std::all_of(value.environment.begin(), value.environment.end(), [](const auto & field) {
             return !field.empty() && field.size() <= 96 &&
                    std::all_of(field.begin(), field.end(), [](char c) {
                      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || c == '-' ||
                             c == '.';
      });
    });
}

bool number(const std::string & text, uint32_t & value)
{
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return !text.empty() && result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

std::string file_identity(const char * path)
{
  struct stat status;
  if (stat(path, &status) != 0) {return "";}
  return std::to_string(status.st_dev) + "." + std::to_string(status.st_ino);
}

std::string read_uuid(const char * path)
{
  std::ifstream input(path);
  std::string result;
  std::getline(input, result);
  result.erase(std::remove(result.begin(), result.end(), '-'), result.end());
  return hex_id(result) ? result : "";
}

uint32_t test_override(const char * name, uint32_t fallback)
{
  const char * value = std::getenv(name);
  if (!value || !*value) {return fallback;}
  uint32_t parsed;
  if (number(value, parsed)) {return parsed;}
  RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend", "Ignoring invalid %s override", name);
  return fallback;
}
}  // namespace

std::string serialize_ipc_metadata(const IpcMetadata & metadata)
{
  if (!metadata.capable || !valid(metadata)) {return std::string(kVersion) + "|0";}
  std::string output = std::string(kVersion) + "|1";
  for (const auto & field : metadata.environment) {
    output += '|' + field;
  }
  return output + '|' + metadata.process + '|' + metadata.gpu + '|' +
         std::to_string(metadata.device) + '|' + std::to_string(metadata.uid);
}

std::optional<IpcMetadata> parse_ipc_metadata(const std::string & wire)
{
  if (wire.size() > kMaxMetadataSize) {return std::nullopt;}
  std::vector<std::string> fields;
  size_t begin = 0;
  do {
    const auto end = wire.find('|', begin);
    fields.push_back(wire.substr(begin, end == std::string::npos ? end : end - begin));
    if (fields.size() > 12) {return std::nullopt;}
    if (end == std::string::npos) {break;}
    begin = end + 1;
  } while (true);
  if (fields.size() == 2 && fields[0] == kVersion && fields[1] == "0") {
    return IpcMetadata{};
  }
  if (fields.size() != 12 || fields[0] != kVersion || fields[1] != "1") {
    return std::nullopt;
  }
  IpcMetadata result;
  result.capable = true;
  std::copy_n(fields.begin() + 2, result.environment.size(), result.environment.begin());
  result.process = fields[8];
  result.gpu = fields[9];
  if (!number(fields[10], result.device) || !number(fields[11], result.uid) || !valid(result)) {
    return std::nullopt;
  }
  return result;
}

IpcDecision decide_ipc(const IpcMetadata & local, const IpcMetadata & peer)
{
  if (!local.capable || !peer.capable || !valid(local) || !valid(peer) ||
    local.environment != peer.environment || local.gpu != peer.gpu ||
    local.device != peer.device || local.uid != peer.uid)
  {
    return {};
  }
  // Ordinals must also match because the existing descriptor/import protocol
  // uses them. GPU UUID prevents equal ordinals for different physical devices
  // from being approved. Supporting remapped ordinals needs a protocol change.
  return {true, local.process == peer.process};
}

IpcMetadata collect_ipc_metadata()
{
  IpcMetadata result;
  try {
    const auto pool = CudaBufferImpl<uint8_t>::get_or_create_global_pool();
    if (!pool || !pool->is_ipc_capable()) {return result;}
    CUuuid uuid;
    CUdevice device;
    if (cuDeviceGet(&device, pool->get_device_id()) != CUDA_SUCCESS ||
      cuDeviceGetUuid_v2(&uuid, device) != CUDA_SUCCESS)
    {
      return result;
    }
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char byte : uuid.bytes) {
      result.gpu += hex[byte >> 4];
      result.gpu += hex[byte & 15];
    }
    result.environment = {read_uuid("/proc/sys/kernel/random/boot_id"),
      file_identity("/dev/shm"), file_identity("/proc/self/ns/net"),
      file_identity("/proc/self/ns/pid"), file_identity("/proc/self/ns/user"),
      file_identity("/proc/self/ns/ipc")};
    // Shared across backend instances in this DSO, independent of PID reuse.
    static const auto process = read_uuid("/proc/sys/kernel/random/uuid");
    result.process = process;
    result.device = test_override("CUDA_BUFFER_DEVICE_ID_OVERRIDE", pool->get_device_id());
    result.uid = test_override("CUDA_BUFFER_UID_OVERRIDE", geteuid());
    result.capable = valid(result);
  } catch (const CudaError &) {
    // Advertising unavailable CUDA must work on machines without a usable GPU.
    return {};
  }
  return result;
}

}  // namespace detail
}  // namespace cuda_buffer_backend
