# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Allocate and access CUDA-backed ROS 2 buffers with scoped handles."""

from collections.abc import Sized
from typing import Iterable
from typing import Optional
from typing import Union

from cuda_buffer._cuda_buffer_py import _allocate_buffer
from cuda_buffer._cuda_buffer_py import _from_input_buffer
from cuda_buffer._cuda_buffer_py import _from_input_cpu_data
from cuda_buffer._cuda_buffer_py import _from_output_buffer
from cuda_buffer._cuda_buffer_py import _to_buffer
from cuda_buffer._cuda_buffer_py import CudaReadHandle
from cuda_buffer._cuda_buffer_py import CudaWriteHandle
from rosidl_buffer import Buffer


class CudaBuffer:
    """Allocate, acquire input/output handles, and copy into CUDA buffers."""

    @staticmethod
    def allocate_buffer(size: int) -> Buffer:
        """Allocate an uninitialized CUDA-backed buffer without synchronizing."""
        return _allocate_buffer(size)

    @staticmethod
    def to_buffer(
        source: Union[bytes, bytearray, memoryview, Iterable[int], int],
        byte_count: int,
        handle: CudaWriteHandle,
        stream: Optional[int] = None,
    ) -> None:
        """
        Copy bytes into an existing, open write handle without finalizing it.

        Host bytes or an iterable of byte values are copied host-to-device;
        an integer source is a device pointer and is copied device-to-device.
        ``byte_count`` must be nonnegative and fit the destination and, for
        host data, the source. Device pointer capacity is the caller's
        responsibility. A null device pointer is allowed only for a zero-byte
        copy.

        ``stream`` is an integer CUDA stream pointer. Omit it to use the
        handle's stream; an explicit stream must match the handle's stream.
        Host copies synchronize that stream before returning, so the source
        can immediately be changed or released. Device copies are asynchronous:
        make the source ready on this stream and retain its storage until the
        copy completes (e.g. use a read handle on the same stream).

        Further GPU writes may be queued on this stream in the same handle
        scope. Only closing the handle finalizes the producer payload.
        """
        if not isinstance(source, (bytes, int)):
            source = bytes(source)
        _to_buffer(source, byte_count, handle, stream)

    @staticmethod
    def from_output_buffer(
        buffer: Union[Buffer, bytes, bytearray, Iterable[int]],
        stream: Optional[int] = None,
    ) -> CudaWriteHandle:
        """
        Acquire scoped write access to output data on a CUDA stream.

        A non-CUDA input supplies the output size only and is replaced by the
        CUDA-backed ``handle.buffer``. ``stream`` is a CUDA stream pointer
        represented as an integer. When omitted, the backend's internal
        stream is used.
        """
        if not isinstance(buffer, Buffer) or buffer.backend_type != 'cuda':
            if isinstance(buffer, Sized):
                size = len(buffer)
            else:
                size = len(bytes(buffer))
            buffer = _allocate_buffer(size)
        return _from_output_buffer(buffer, stream)

    @staticmethod
    def from_input_buffer(
        buffer: Union[Buffer, bytes, bytearray, Iterable[int]],
        stream: Optional[int] = None,
    ) -> CudaReadHandle:
        """
        Acquire scoped read access to input data on a CUDA stream.

        Non-CUDA input is promoted to a CUDA buffer and retained by the
        returned handle. ``stream`` is a CUDA stream pointer represented as
        an integer. When omitted, the backend's internal stream is used.
        """
        if isinstance(buffer, Buffer):
            return _from_input_buffer(buffer, stream)
        if not isinstance(buffer, bytes):
            buffer = bytes(buffer)
        return _from_input_cpu_data(buffer, stream)


__all__ = ['CudaBuffer', 'CudaReadHandle', 'CudaWriteHandle']
