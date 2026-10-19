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

from typing import Iterable
from typing import Optional
from typing import Union

from cuda_buffer._cuda_buffer_py import _allocate_buffer
from cuda_buffer._cuda_buffer_py import _from_input_buffer
from cuda_buffer._cuda_buffer_py import _from_input_cpu_data
from cuda_buffer._cuda_buffer_py import _from_output_buffer
from cuda_buffer._cuda_buffer_py import _from_output_cpu_data
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
        buffer: Union[Buffer, bytes, bytearray, memoryview, Iterable[int]],
        stream: Optional[int] = None,
    ) -> CudaWriteHandle:
        """
        Acquire scoped write access to output data on a CUDA stream.

        The bindings promote non-CUDA buffers using their byte size, without
        copying their contents. Retrieve the Python-owned replacement with
        ``handle.get_promoted_buffer()`` before closing the handle; it returns
        None for an existing CUDA buffer. The original argument is unchanged.
        Containers use the byte sequence produced by bytes(buffer).
        ``stream`` is an integer CUDA stream pointer; None uses the internal stream.
        """
        if isinstance(buffer, Buffer):
            return _from_output_buffer(buffer, stream)
        if not isinstance(buffer, bytes):
            buffer = bytes(buffer)
        return _from_output_cpu_data(buffer, stream)

    @staticmethod
    def from_input_buffer(
        buffer: Union[Buffer, bytes, bytearray, memoryview, Iterable[int]],
        stream: Optional[int] = None,
    ) -> CudaReadHandle:
        """
        Acquire scoped read access to input data on a CUDA stream.

        The bindings promote non-CUDA input to a Python-owned CUDA buffer,
        copying its contents through an internal producer write scope. The copy
        completes before returning, so the host source may be changed or released.
        ``handle.get_promoted_buffer()`` returns the replacement, or None for
        an existing CUDA buffer. Retrieve it before closing the handle to retain
        it independently. ``stream`` is an integer CUDA stream pointer; None
        uses the internal stream.
        """
        if isinstance(buffer, Buffer):
            return _from_input_buffer(buffer, stream)
        if not isinstance(buffer, bytes):
            buffer = bytes(buffer)
        return _from_input_cpu_data(buffer, stream)


__all__ = ['CudaBuffer', 'CudaReadHandle', 'CudaWriteHandle']
