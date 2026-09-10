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

#ifndef CUDA_BUFFER__CUDA_BUFFER_API_HPP_
#define CUDA_BUFFER__CUDA_BUFFER_API_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/external_memory_pool.hpp"
#include "cuda_buffer/transport_pool.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace cuda_buffer_backend
{

/// \brief Allocate \p count elements out of a caller-owned device region.
///
/// The counterpart to allocate_buffer() for a process that already has a device
/// allocator of its own and wants ROS payloads to come out of *it* -- a
/// middleware whose transport memory is an NvSciBuf pool being the case this
/// was built for. See ExternalMemoryPool::import_nvscibuf().
///
/// The returned buffer is a normal CUDA-backed rosidl::Buffer in every respect
/// an application can observe: backend \c "cuda", the same read and write
/// handles, the same event ordering, and no promotion or host round trip in
/// from_input_buffer() / from_output_buffer(). It differs from allocate_buffer()
/// only in where the bytes live and who reclaims them -- the block returns to
/// \p pool when the buffer dies, and the pool's region is never freed by this
/// backend.
///
/// Unlike adopt_buffer(), the result *can* be resized: growth reallocates
/// within the same pool, which needs room for the old and new blocks at once
/// because the copy happens before the old one is released.
///
/// \param pool The region to allocate from. Kept alive by the returned buffer.
/// \param count Elements, not bytes.
/// \throw CudaError if \p pool is null or has no contiguous run left for
///   \p count elements. \c count == 0 returns an empty buffer and allocates
///   nothing.
template<typename T = uint8_t>
rosidl::Buffer<T> allocate_buffer_from(
  std::shared_ptr<ExternalMemoryPool> pool, size_t count)
{
  if (!pool) {
    throw CudaError("allocate_buffer_from called with a null pool");
  }
  return rosidl::Buffer<T>(
    std::make_unique<CudaBufferImpl<T>>(std::move(pool), count));
}

/// \brief Allocate a fresh CUDA-backed \c rosidl::Buffer<uint8_t> of \p count
/// bytes. Pure allocation: no handle is acquired, no data is copied. The
/// caller owns the returned buffer and assigns it wherever the schema needs
/// it (e.g. `msg.data = allocate_buffer(n)` for messages that follow the
/// `data` convention, or an arbitrary field on other messages).
///
/// Which device memory depends on the RMW the process actually loaded. One
/// that offers a participant-wide transport pool is asked first, so the bytes
/// are written straight into memory the transport can send without copying
/// them; every other RMW falls through to this process's own CUDA pool, which
/// is what this function has always done. The two are indistinguishable
/// afterwards -- backend \c "cuda", the same handles, the same event ordering,
/// resizable within whichever pool they came from -- and that is why the
/// zero-copy path was put here rather than behind a second entry point: no
/// call site has to be rewritten, or even to know.
///
/// Falling back is never silent about being *wrong*, only about being ordinary:
/// once a transport has handed over a pool, failing to satisfy the request out
/// of it throws rather than quietly costing a copy.
///
/// Ordered after allocate_buffer_from() because it calls it, and after the
/// include of transport_pool.hpp because it calls that. The include runs one
/// way only: transport_pool.hpp must not include this header back.
inline rosidl::Buffer<uint8_t> allocate_buffer(size_t count)
{
  // Zero elements are worth nobody's transport memory, and 0 is the protocol's
  // "no hint" besides -- asking a provider to pick a size class for an
  // allocation that will not happen.
  if (count > 0) {
    if (std::shared_ptr<ExternalMemoryPool> pool = shared_transport_pool(count)) {
      return allocate_buffer_from<uint8_t>(std::move(pool), count);
    }
  }
  return rosidl::Buffer<uint8_t>(
    std::make_unique<CudaBufferImpl<uint8_t>>(count));
}

/// \brief Wrap \p count elements of device memory allocated somewhere else.
///
/// The counterpart to allocate_buffer() for memory this backend did not create:
/// a zero-copy transport's shared slot, an NvSciBuf attachment imported into
/// CUDA, a graphics interop surface. The returned buffer reports backend
/// \c "cuda" and is indistinguishable from an allocated one at every call site
/// that matters -- from_input_buffer(), from_output_buffer(), to_vector(),
/// clone(), the read and write handles and their event ordering all behave
/// identically. An application never has to know which kind it was handed.
///
/// Two things it will not do, both because the storage is on loan: it never
/// frees the memory, and it never reallocates it. resize() therefore narrows
/// in place and refuses to grow (see shrink_buffer()).
///
/// \param device_ptr Device address, already mapped into this process's CUDA
///   context by whoever owns it.
/// \param count Elements, not bytes.
/// \param keepalive Released once the GPU is finished with the memory -- after
///   every read and write event on it has been synchronized, not merely when
///   the buffer is dropped. Pass whatever the owner uses to mean "still in
///   use". Null is legal when the storage outlives the buffer by construction.
/// \param device_id CUDA device the pointer belongs to; -1 asks the current
///   device, which is right whenever the caller mapped the memory itself.
/// \throw CudaError if \p device_ptr is null, \p count is zero, or
///   \p device_id is -1 and no current device can be determined.
template<typename T = uint8_t>
rosidl::Buffer<T> adopt_buffer(
  void * device_ptr, size_t count, std::shared_ptr<void> keepalive = nullptr,
  int device_id = -1)
{
  if (device_ptr == nullptr) {
    throw CudaError("adopt_buffer called with a null device pointer");
  }
  if (count == 0) {
    throw CudaError("adopt_buffer called with a zero element count");
  }
  if (device_id < 0) {
    int current = 0;
    CUDA_CHECK(cudaGetDevice(&current));
    device_id = current;
  }

  CudaBuffer wrapper = CudaBuffer::adopt(
    device_ptr, count * sizeof(T), device_id, std::move(keepalive));

  // The same event an allocated buffer gets. Without it the handles still hand
  // out the right pointer but cannot order a reader behind a writer, which is
  // the entire reason a producer can fill this asynchronously and publish
  // without synchronizing.
  if (cudaEvent_t ev = make_buffer_write_event()) {
    wrapper.set_write_event(ev, true);
  }

  return rosidl::Buffer<T>(
    std::make_unique<CudaBufferImpl<T>>(std::move(wrapper), count));
}

/// \brief Narrow \p buffer to \p count elements without reallocating.
///
/// For a producer that was handed fixed-size storage -- a transport loan is the
/// motivating case -- and filled less than all of it. There is no other way to
/// say so: rosidl::Buffer::resize() throws for every non-CPU backend, so such a
/// producer would otherwise have to publish its whole capacity.
///
/// Neither reallocates nor copies for either kind of CUDA buffer, adopted or
/// pooled; only the reported count changes.
///
/// \return true if \p buffer is CUDA-backed and was narrowed. False leaves it
///         untouched -- a CPU-backed buffer can simply be resized, and any
///         other backend is not ours to narrow.
template<typename T>
bool shrink_buffer(rosidl::Buffer<T> & buffer, size_t count)
{
  auto * cuda_impl = dynamic_cast<CudaBufferImpl<T> *>(buffer.get_impl());
  return cuda_impl != nullptr && cuda_impl->shrink(count);
}

namespace detail
{

/// \brief Heap-allocate a fresh CUDA-backed rosidl::Buffer<uint8_t>, held via
/// shared_ptr so it can ride along with a promoted read/write handle.
inline std::shared_ptr<rosidl::Buffer<uint8_t>> allocate_cuda_buffer_shared(
  size_t byte_count)
{
  auto cuda_impl = std::make_unique<CudaBufferImpl<uint8_t>>(byte_count);
  return std::make_shared<rosidl::Buffer<uint8_t>>(std::move(cuda_impl));
}

inline CudaBufferImpl<uint8_t> * cuda_impl_of(rosidl::Buffer<uint8_t> & buffer)
{
  return dynamic_cast<CudaBufferImpl<uint8_t> *>(buffer.get_impl());
}

}  // namespace detail

/// \brief Acquire a write handle for a CUDA-backed buffer.
/// \details If \p buffer is already CUDA-backed, returns a write handle
/// directly. If the buffer is non-CUDA (e.g. CPU-backed), a fresh
/// CUDA-backed \c rosidl::Buffer<uint8_t> is allocated (no H2D copy; the
/// caller is about to overwrite it) and a write handle for the new buffer
/// is returned. The promoted buffer is attached to the handle via
/// \c WriteHandle::get_promoted_buffer() so the caller can substitute the
/// buffer back into the message they're publishing.
template<typename T>
WriteHandle from_output_buffer(
  rosidl::Buffer<T> & buffer,
  cudaStream_t stream)
{
  auto * impl = buffer.get_impl();
  if (!impl) {
    throw CudaError("from_output_buffer called on buffer with null implementation");
  }
  if (buffer.size() == 0) {
    throw CudaError("from_output_buffer called on empty buffer");
  }
  auto * cuda_impl = dynamic_cast<CudaBufferImpl<T> *>(impl);
  if (cuda_impl) {
    cuda_impl->set_stream(stream);
    return cuda_impl->get_cuda_buffer().get_write_handle(stream);
  }

  size_t byte_count = buffer.size() * sizeof(T);
  auto promoted = detail::allocate_cuda_buffer_shared(byte_count);
  auto * promoted_impl = detail::cuda_impl_of(*promoted);
  promoted_impl->set_stream(stream);
  auto wh = promoted_impl->get_cuda_buffer().get_write_handle(stream);
  wh.set_promoted_buffer(std::move(promoted));
  return wh;
}

/// \brief Acquire a read handle for a CUDA-backed buffer.
/// \details If \p buffer is already CUDA-backed, returns a read handle
/// directly. If the buffer is non-CUDA (e.g. CPU-backed), a new CUDA-backed
/// \c rosidl::Buffer<uint8_t> is allocated, the source contents are copied
/// host-to-device, and a read handle for the new buffer is returned.
template<typename T>
ReadHandle from_input_buffer(
  const rosidl::Buffer<T> & buffer,
  cudaStream_t stream)
{
  const auto * impl = buffer.get_impl();
  if (!impl) {
    throw CudaError("from_input_buffer called on buffer with null implementation");
  }
  if (buffer.size() == 0) {
    throw CudaError("from_input_buffer called on empty buffer");
  }
  const auto * cuda_impl = dynamic_cast<const CudaBufferImpl<T> *>(impl);
  if (cuda_impl) {
    return cuda_impl->get_cuda_buffer().get_read_handle(stream);
  }

  size_t byte_count = buffer.size() * sizeof(T);
  auto promoted = detail::allocate_cuda_buffer_shared(byte_count);
  auto * promoted_impl = detail::cuda_impl_of(*promoted);
  {
    auto wh = promoted_impl->get_cuda_buffer().get_write_handle(stream);
    if (impl->get_backend_type() == "cpu") {
      // Host storage, and it outlives this call, so copy straight out of it.
      CUDA_CHECK(cudaMemcpyAsync(
        wh.get_ptr(), buffer.data(), byte_count, cudaMemcpyHostToDevice, stream));
    } else {
      // Some other non-CPU backend: another accelerator, or a buffer delivered
      // by a transport. data() throws for every one of them -- it is CPU-only
      // by contract -- so the portable way in is rosidl_buffer's explicit
      // conversion. Reaching here at all means a copy was already unavoidable.
      //
      // The staging vector is local, so the copy has to land before it goes
      // away; that is what the synchronize is for, and why it is only on this
      // branch.
      const std::vector<T> host = buffer.to_vector();
      CUDA_CHECK(cudaMemcpyAsync(
        wh.get_ptr(), host.data(), byte_count, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  auto rh = promoted_impl->get_cuda_buffer().get_read_handle(stream);
  rh.set_promoted_buffer(std::move(promoted));
  return rh;
}

/// \brief Copy \p byte_count bytes from \p src into the memory referenced by \p wh
/// using \p stream and the given \p kind. Does not allocate.
inline void to_buffer(
  const void * src,
  size_t byte_count,
  WriteHandle & wh,
  cudaStream_t stream,
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice)
{
  if (byte_count == 0 || !src || !wh.get_ptr()) {
    return;
  }
  CUDA_CHECK(cudaMemcpyAsync(wh.get_ptr(), src, byte_count, kind, stream));
}

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_API_HPP_
