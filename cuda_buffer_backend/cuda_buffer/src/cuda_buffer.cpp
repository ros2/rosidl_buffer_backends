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

#include "cuda_buffer/cuda_buffer.hpp"

#include <memory>
#include <mutex>

#include "cuda_buffer/vmm_buffer.hpp"

namespace cuda_buffer_backend
{

// -- CudaBuffer --

CudaBuffer::~CudaBuffer() = default;

std::unique_ptr<CudaBuffer> CudaBuffer::allocate(size_t byte_size)
{
  // CUDA VMM is the only storage backend today. Other backends plug in here.
  return VmmBuffer::allocate(byte_size);
}

ReadHandle CudaBuffer::get_read_handle(cudaStream_t stream) const
{
  (void)reap_completed_reads();

  if (handle_state_) {
    std::lock_guard<std::mutex> lk(handle_state_->mtx);
    if (handle_state_->state == HandleState::State::InUse) {
      finalize_write_handle_locked();
    }
  }

  wait_for_write(stream);

  return ReadHandle(device_ptr(), this, stream);
}

WriteHandle CudaBuffer::get_write_handle(cudaStream_t stream)
{
  std::lock_guard<std::mutex> lg(write_acquire_mutex_);
  const size_t pending_reads = reap_completed_reads();

  if (handle_state_) {
    std::lock_guard<std::mutex> lk(handle_state_->mtx);
    if (handle_state_->state == HandleState::State::InUse) {
      throw CudaError("CudaBuffer: write handle already in use; cannot acquire second handle");
    }
    if (handle_state_->state == HandleState::State::Finalized) {
      throw CudaError("CudaBuffer: write already finalized; cannot acquire write handle");
    }
  }

  if (pending_reads > 0) {
    throw CudaError("CudaBuffer: read events exist; cannot re-acquire write handle");
  }

  if (!handle_state_) {handle_state_ = std::make_shared<HandleState>();}
  handle_state_->state = HandleState::State::InUse;
  handle_state_->write_stream = stream;
  return WriteHandle(device_ptr(), this, stream, handle_state_);
}

void CudaBuffer::finalize_write_handle() const
{
  if (handle_state_ == nullptr) {return;}
  std::lock_guard<std::mutex> lock(handle_state_->mtx);
  finalize_write_handle_locked();
}

void CudaBuffer::finalize_write_handle_locked() const
{
  if (handle_state_->state != HandleState::State::InUse) {return;}
  if (cuda_is_stream_usable(handle_state_->write_stream)) {
    signal_write(handle_state_->write_stream);
  }
  handle_state_->state = HandleState::State::Finalized;
  handle_state_->write_stream = nullptr;
}

// -- ReadHandle --

void ReadHandle::release() noexcept
{
  if (!owner_ || !cuda_is_stream_usable(stream_)) {return;}
  owner_->signal_read(stream_);
}

// -- WriteHandle --

void WriteHandle::release() noexcept
{
  if (!state_) {return;}
  std::lock_guard<std::mutex> lock(state_->mtx);
  // A finalized write no longer refers to its buffer, which may be gone by now.
  if (state_->state == HandleState::State::Finalized) {return;}
  cudaStream_t ws = state_->write_stream ? state_->write_stream : stream_;
  if (owner_ && cuda_is_stream_usable(ws)) {
    owner_->signal_write(ws);
  }
  state_->state = HandleState::State::Finalized;
  state_->write_stream = nullptr;
}

}  // namespace cuda_buffer_backend
