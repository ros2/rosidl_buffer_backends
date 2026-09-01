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

"""Python factories for CUDA-backed ROS 2 buffers."""

from collections.abc import Sized
from typing import Iterable
from typing import Optional
from typing import Union

from cuda_buffer._cuda_buffer_py import _allocate_buffer
from cuda_buffer._cuda_buffer_py import _from_cpu_data
from cuda_buffer._cuda_buffer_py import _from_input_buffer
from cuda_buffer._cuda_buffer_py import _from_input_cpu_data
from cuda_buffer._cuda_buffer_py import _from_output_buffer
from cuda_buffer._cuda_buffer_py import _from_size
from cuda_buffer._cuda_buffer_py import _get_internal_stream
from cuda_buffer._cuda_buffer_py import CudaReadHandle
from cuda_buffer._cuda_buffer_py import CudaWriteHandle
from rosidl_buffer import Buffer


class CudaBuffer:
    """Create CUDA-backed :class:`rosidl_buffer.Buffer` objects."""

    @staticmethod
    def from_cpu(data: Union[bytes, bytearray, Iterable[int]]) -> Buffer:
        """Copy CPU byte data into a new CUDA-backed buffer."""
        if not isinstance(data, bytes):
            data = bytes(data)
        return _from_cpu_data(data)

    @staticmethod
    def from_size(size: int) -> Buffer:
        """Create a zero-initialized CUDA-backed buffer with ``size`` bytes."""
        return _from_size(size)

    @staticmethod
    def allocate_buffer(size: int) -> Buffer:
        """Allocate an uninitialized CUDA-backed buffer without synchronizing."""
        return _allocate_buffer(size)

    @staticmethod
    def get_internal_stream() -> int:
        """Return the CUDA stream used when no explicit stream is supplied."""
        return _get_internal_stream()

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
