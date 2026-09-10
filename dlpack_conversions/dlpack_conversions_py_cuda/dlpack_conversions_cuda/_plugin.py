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

from dlpack_conversions._dlpack_bridge import buffer_address
from dlpack_conversions._dlpack_bridge import make_dlpack_capsule
from dlpack_conversions._plugin import CUDA
from dlpack_conversions._plugin import StorageRegistry
from dlpack_conversions._plugin import TensorMetadata


_CUDART_CANDIDATES = ('libcudart.so', 'libcudart.so.13', 'libcudart.so.12')
# Unified addressing lets the driver infer the direction of every copy.
_CUDA_MEMCPY_DEFAULT = 4


@lru_cache(maxsize=1)
def _runtime() -> Optional[ctypes.CDLL]:
    for name in _CUDART_CANDIDATES:
        try:
            runtime = ctypes.CDLL(name)
        except OSError:
            continue
        runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaGetDeviceCount.restype = ctypes.c_int
        runtime.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,
        ]
        runtime.cudaMemcpyAsync.restype = ctypes.c_int
        runtime.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        runtime.cudaStreamSynchronize.restype = ctypes.c_int
        return runtime
    return None


@lru_cache(maxsize=1)
def _device_count() -> int:
    runtime = _runtime()
    if runtime is None:
        return 0
    count = ctypes.c_int(0)
    if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0:
        return 0
    return count.value


def _copy(destination: int, source: int, byte_count: int, stream: int) -> None:
    runtime = _runtime()
    result = runtime.cudaMemcpyAsync(
        ctypes.c_void_p(destination), ctypes.c_void_p(source), byte_count,
        _CUDA_MEMCPY_DEFAULT, ctypes.c_void_p(stream))
    if result != 0:
        raise RuntimeError(f'cudaMemcpyAsync failed with error {result}')
    # Callers may read the destination as soon as this returns.
    result = runtime.cudaStreamSynchronize(ctypes.c_void_p(stream))
    if result != 0:
        raise RuntimeError(
            f'cudaStreamSynchronize failed with error {result}')


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
            CudaBuffer.from_input_buffer(data, stream), data, metadata
        )

    def acquire_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> object:
        return _capsule(
            CudaBuffer.from_output_buffer(data, stream), data, metadata
        )

    def copy_to(
        self,
        data: object,
        source: int,
        byte_count: int,
        source_backend: str,
        stream: Optional[int],
    ) -> None:
        del source_backend
        if getattr(data, 'backend_type', 'cpu') != 'cuda':
            _copy(buffer_address(data), source, byte_count, stream or 0)
            return
        with CudaBuffer.from_output_buffer(data, stream) as handle:
            _copy(handle.device_ptr, source, byte_count, stream or 0)

    def unavailable_error(self) -> RuntimeError:
        return RuntimeError('No CUDA device is visible to this process')


class _Lease:
    """
    Hold a mapping open, and the buffer it maps, until DLPack is done.

    A handle that outlives its buffer leaves a dangling mapping, so the
    handle is closed first and only then is the buffer released.
    """

    def __init__(self, handle: object, data: object) -> None:
        self._handle = handle
        self._data = data

    def __del__(self) -> None:
        self._handle.close()


def _capsule(
    handle: object, data: object, metadata: TensorMetadata
) -> object:
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
        _Lease(handle, data),
    )


def register(registry: StorageRegistry) -> None:
    registry.register(CudaStoragePlugin())
