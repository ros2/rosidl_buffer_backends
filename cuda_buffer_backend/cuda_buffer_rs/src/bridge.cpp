// Copyright 2026 Open Source Robotics Foundation, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_buffer_rs/src/bridge.rs.h"

#include <new>
#include <stdexcept>

namespace cuda_buffer_rs
{
namespace
{
template<typename Function>
auto checked(NativeErrorKind & error, Function && function) -> decltype(function())
{
  try {
    return function();
  } catch (const std::bad_alloc &) {
    error = NativeErrorKind::BadAlloc;
    throw;
  } catch (const cuda_buffer_backend::CudaError &) {
    error = NativeErrorKind::Cuda;
    throw;
  } catch (const std::invalid_argument &) {
    error = NativeErrorKind::InvalidArgument;
    throw;
  } catch (const std::exception &) {
    error = NativeErrorKind::Other;
    throw;
  } catch (...) {
    error = NativeErrorKind::Other;
    throw std::runtime_error("unknown native CUDA error");
  }
}

}  // namespace

uintptr_t internal_stream(NativeErrorKind & error)
{
  return checked(error, []() {
             return reinterpret_cast<uintptr_t>(cuda_buffer_backend::get_internal_stream());
    });
}

std::unique_ptr<CxxBuffer> allocate(size_t size, NativeErrorKind & error)
{
  return checked(error, [&]() {
             return std::make_unique<CxxBuffer>(cuda_buffer_backend::allocate_buffer(size));
    });
}

bool is_cuda_backed(const CxxBuffer & buffer) noexcept
{
  return dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    buffer.get_impl()) != nullptr;
}

int32_t device_id(const CxxBuffer & buffer, NativeErrorKind & error)
{
  return checked(error, [&]() {
             const auto * impl = dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
               buffer.get_impl());
             if (!impl) {
               throw std::invalid_argument("expected a CUDA-backed buffer");
             }
             return impl->get_device_id();
    });
}

std::unique_ptr<CxxReadHandle> acquire_read(
  const CxxBuffer & buffer, uintptr_t stream, NativeErrorKind & error)
{
  return checked(error, [&]() {
             return std::make_unique<CxxReadHandle>(
        cuda_buffer_backend::from_input_buffer(buffer, reinterpret_cast<cudaStream_t>(stream)),
        buffer.size());
    });
}

std::unique_ptr<CxxWriteHandle> acquire_write(
  CxxBuffer & buffer, uintptr_t stream, NativeErrorKind & error)
{
  return checked(error, [&]() {
             auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
             return std::make_unique<CxxWriteHandle>(
        cuda_buffer_backend::from_output_buffer(buffer, cuda_stream), buffer.size());
    });
}

void CxxWriteHandle::copy_from(
  const uint8_t * source, size_t size, uintptr_t stream, int32_t kind)
{
  cuda_buffer_backend::to_buffer(
    source, size, handle_, reinterpret_cast<cudaStream_t>(stream), static_cast<cudaMemcpyKind>(kind));
}

void copy_to_buffer(
  CxxWriteHandle & handle, const uint8_t * source, size_t size,
  uintptr_t stream, int32_t kind, NativeErrorKind & error)
{
  checked(error, [&]() {handle.copy_from(source, size, stream, kind);});
}
}  // namespace cuda_buffer_rs
