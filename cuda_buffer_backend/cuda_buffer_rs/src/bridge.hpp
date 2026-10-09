// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDA_BUFFER_RS__BRIDGE_HPP_
#define CUDA_BUFFER_RS__BRIDGE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rosidl_runtime_rs/src/buffer_bridge.hpp"

namespace cuda_buffer_rs
{
enum class NativeErrorKind : uint8_t;
using rosidl_buffer_rs::CxxBuffer;

class CxxReadHandle
{
public:
  CxxReadHandle(cuda_buffer_backend::ReadHandle && handle, size_t size)
  : handle_(std::move(handle)), size_(size) {}
  const uint8_t * data() const noexcept {return handle_.get_ptr();}
  size_t size() const noexcept {return size_;}

private:
  cuda_buffer_backend::ReadHandle handle_;
  size_t size_;
};

class CxxWriteHandle
{
public:
  CxxWriteHandle(cuda_buffer_backend::WriteHandle && handle, size_t size)
  : handle_(std::move(handle)), data_(handle_.get_ptr()), size_(size) {}
  uint8_t * data() const noexcept {return data_;}
  size_t size() const noexcept {return size_;}
  void copy_from(const uint8_t * source, size_t size, uintptr_t stream, int32_t kind);

private:
  cuda_buffer_backend::WriteHandle handle_;
  uint8_t * data_;
  size_t size_;
};

uintptr_t internal_stream(NativeErrorKind & error);
std::unique_ptr<CxxBuffer> allocate(size_t size, NativeErrorKind & error);
bool is_cuda_backed(const CxxBuffer & buffer) noexcept;
int32_t device_id(const CxxBuffer & buffer, NativeErrorKind & error);
std::unique_ptr<CxxReadHandle> acquire_read(
  const CxxBuffer & buffer, uintptr_t stream, NativeErrorKind & error);
std::unique_ptr<CxxWriteHandle> acquire_write(
  CxxBuffer & buffer, uintptr_t stream, NativeErrorKind & error);
void copy_to_buffer(
  CxxWriteHandle & handle, const uint8_t * source, size_t size,
  uintptr_t stream, int32_t kind, NativeErrorKind & error);
}  // namespace cuda_buffer_rs

#endif  // CUDA_BUFFER_RS__BRIDGE_HPP_
