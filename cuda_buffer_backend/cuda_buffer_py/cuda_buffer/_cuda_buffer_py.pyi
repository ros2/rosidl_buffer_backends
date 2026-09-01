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

from types import TracebackType
from typing import Optional
from typing import Type

from rosidl_buffer import Buffer


class CudaReadHandle:
    @property
    def device_ptr(self) -> int: ...
    @property
    def device_id(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    def get_ptr(self) -> int: ...
    def close(self) -> None: ...
    def __enter__(self) -> CudaReadHandle: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool: ...


class CudaWriteHandle:
    @property
    def device_ptr(self) -> int: ...
    @property
    def device_id(self) -> int: ...
    @property
    def buffer(self) -> Buffer: ...
    @property
    def closed(self) -> bool: ...
    def get_ptr(self) -> int: ...
    def close(self) -> None: ...
    def __enter__(self) -> CudaWriteHandle: ...
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool: ...


def _from_cpu_data(data: bytes) -> Buffer: ...
def _get_internal_stream() -> int: ...
def _from_size(size: int) -> Buffer: ...
def _allocate_buffer(size: int) -> Buffer: ...
def _from_output_buffer(
    buffer: Buffer,
    stream: Optional[int] = None,
) -> CudaWriteHandle: ...
def _from_input_buffer(
    buffer: Buffer,
    stream: Optional[int] = None,
) -> CudaReadHandle: ...
def _from_input_cpu_data(
    data: bytes,
    stream: Optional[int] = None,
) -> CudaReadHandle: ...
