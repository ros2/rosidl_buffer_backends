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

from array import array
from typing import Optional

from dlpack_conversions._dlpack_bridge import buffer_address
from dlpack_conversions._dlpack_bridge import make_dlpack_capsule
from dlpack_conversions._plugin import CPU
from dlpack_conversions._plugin import StorageRegistry
from dlpack_conversions._plugin import TensorMetadata


class CpuStoragePlugin:
    """Host memory storage for DLPack tensor messages."""

    backends = ('cpu',)
    dl_device_types = (CPU,)
    priority = 0
    is_fallback = True

    def is_available(self) -> bool:
        return True

    def matches(self, data: object) -> bool:
        try:
            memoryview(data)
        except TypeError:
            return False
        return True

    def allocate(self, byte_count: int, backend: str) -> array:
        del backend
        return array('B', bytes(byte_count))

    def acquire_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> object:
        del stream
        return _capsule(data, metadata)

    def acquire_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> object:
        del stream
        return _capsule(data, metadata)

    def unavailable_error(self) -> RuntimeError:
        return RuntimeError('Host memory storage is unavailable')


def _capsule(data: object, metadata: TensorMetadata) -> object:
    # DLPack consumers disagree on byte_offset, so fold it into the pointer.
    return make_dlpack_capsule(
        buffer_address(data) + metadata.byte_offset,
        CPU,
        0,
        metadata.dtype_code,
        metadata.dtype_bits,
        metadata.dtype_lanes,
        list(metadata.shape),
        list(metadata.strides),
        0,
        data,
    )


def register(registry: StorageRegistry) -> None:
    registry.register(CpuStoragePlugin())
