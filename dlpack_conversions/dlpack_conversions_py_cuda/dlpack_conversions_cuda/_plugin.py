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

import ctypes
from functools import lru_cache
from typing import Optional

from cuda_buffer import CudaBuffer

from dlpack_conversions._dlpack_bridge import make_dlpack_capsule
from dlpack_conversions._plugin import CUDA
from dlpack_conversions._plugin import StorageRegistry
from dlpack_conversions._plugin import TensorMetadata


_CUDART_CANDIDATES = ('libcudart.so', 'libcudart.so.13', 'libcudart.so.12')


@lru_cache(maxsize=1)
def _device_count() -> int:
    for name in _CUDART_CANDIDATES:
        try:
            runtime = ctypes.CDLL(name)
        except OSError:
            continue
        count = ctypes.c_int(0)
        if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0:
            return 0
        return count.value
    return 0


class CudaStoragePlugin:
    """CUDA device memory storage for DLPack tensor messages."""

    backends = ('cuda',)
    dl_device_types = (CUDA,)
    priority = 100

    def is_available(self) -> bool:
        return _device_count() > 0

    def matches(self, data: object) -> bool:
        del data
        return False

    def allocate(self, byte_count: int, backend: str) -> object:
        del backend
        return CudaBuffer.allocate_buffer(byte_count)

    def acquire_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> object:
        return _capsule(
            CudaBuffer.from_input_buffer(data, stream), metadata
        )

    def acquire_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> object:
        return _capsule(
            CudaBuffer.from_output_buffer(data, stream), metadata
        )

    def unavailable_error(self) -> RuntimeError:
        return RuntimeError('No CUDA device is visible to this process')


def _capsule(handle: object, metadata: TensorMetadata) -> object:
    # DLPack consumers disagree on byte_offset, so fold it into the pointer.
    return make_dlpack_capsule(
        handle.device_ptr + metadata.byte_offset,
        CUDA,
        handle.device_id,
        metadata.dtype_code,
        metadata.dtype_bits,
        metadata.dtype_lanes,
        list(metadata.shape),
        list(metadata.strides),
        0,
        handle,
    )


def register(registry: StorageRegistry) -> None:
    registry.register(CudaStoragePlugin())
