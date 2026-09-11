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
from typing import Optional

from cuda_buffer import CudaBuffer
import onnxruntime as ort

from onnxruntime_conversions._ort_bridge import make_dlpack_capsule
from onnxruntime_conversions._plugin import ConversionRegistry
from onnxruntime_conversions._plugin import DLPackProducer
from onnxruntime_conversions._plugin import TensorMetadata


_CUDA = ctypes.CDLL('libcudart.so')
_CUDA.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
_CUDA.cudaGetDeviceCount.restype = ctypes.c_int
_CUDA.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_void_p,
]
_CUDA.cudaMemcpyAsync.restype = ctypes.c_int
_CUDA.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_CUDA.cudaStreamSynchronize.restype = ctypes.c_int

_HOST_TO_DEVICE = 1
_DEVICE_TO_HOST = 2
_DEVICE_TO_DEVICE = 3


def _check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f'{operation} failed with CUDA error {result}')


class _Lease:
    """Keep the CUDA mapping and backing buffer alive with the OrtValue."""

    def __init__(self, handle: object, data: object) -> None:
        self._handle = handle
        self._data = data

    def __del__(self) -> None:
        self._handle.close()


class CudaConversionPlugin:
    """Direct OrtValue views over CUDA-backed tensor message storage."""

    backends = ('cuda',)
    device_types = ('cuda',)
    priority = 100

    def is_available(self) -> bool:
        count = ctypes.c_int()
        return _CUDA.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0

    def matches(self, data: object) -> bool:
        del data
        return False

    def allocate(self, byte_count: int, backend: str) -> object:
        del backend
        return CudaBuffer.allocate_buffer(byte_count)

    @staticmethod
    def _value(
        handle: object, data: object, metadata: TensorMetadata,
    ) -> tuple[object, object]:
        lease = _Lease(handle, data)
        dtype_code = 1 if metadata.element_type == 9 else metadata.dtype_code
        capsule = make_dlpack_capsule(
            handle.device_ptr + metadata.byte_offset,
            2,
            handle.device_id,
            dtype_code,
            metadata.dtype_bits,
            metadata.dtype_lanes,
            list(metadata.shape),
            list(metadata.strides),
            0,
            lease,
        )
        producer = DLPackProducer(
            capsule, metadata.numpy_dtype, (2, handle.device_id))
        return ort.OrtValue.from_dlpack(producer), data

    def from_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> tuple[object, object]:
        return self._value(
            CudaBuffer.from_input_buffer(data, stream), data, metadata)

    def from_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> tuple[object, object]:
        return self._value(
            CudaBuffer.from_output_buffer(data, stream), data, metadata)

    def copy_to(
        self,
        data: object,
        metadata: TensorMetadata,
        source: object,
        stream: Optional[int],
    ) -> None:
        cuda_stream = ctypes.c_void_p(stream or 0)
        source_is_cpu = source.device_name().lower() == 'cpu'
        if hasattr(data, 'backend_type') and data.backend_type == 'cuda':
            with CudaBuffer.from_output_buffer(data, stream) as handle:
                source_array = source.numpy() if source_is_cpu else None
                source_pointer = (
                    source_array.ctypes.data if source_is_cpu
                    else source.data_ptr()
                )
                kind = _HOST_TO_DEVICE if source_is_cpu else _DEVICE_TO_DEVICE
                _check(_CUDA.cudaMemcpyAsync(
                    ctypes.c_void_p(handle.device_ptr + metadata.byte_offset),
                    ctypes.c_void_p(source_pointer), metadata.byte_count,
                    kind, cuda_stream), 'cudaMemcpyAsync')
                if source_is_cpu:
                    _check(_CUDA.cudaStreamSynchronize(cuda_stream),
                           'cudaStreamSynchronize')
            return

        if source_is_cpu:
            raise RuntimeError(
                'CUDA plugin expected a CUDA source for a CPU destination')
        destination = (ctypes.c_ubyte * len(data)).from_buffer(data)
        _check(_CUDA.cudaMemcpyAsync(
            ctypes.cast(destination, ctypes.c_void_p),
            ctypes.c_void_p(source.data_ptr()), metadata.byte_count,
            _DEVICE_TO_HOST, cuda_stream), 'cudaMemcpyAsync')
        _check(_CUDA.cudaStreamSynchronize(cuda_stream),
               'cudaStreamSynchronize')

    def session_providers(
        self, device_id: int, stream: Optional[int]
    ) -> list:
        if stream is None:
            raise ValueError(
                "Backend 'cuda' requires an explicit execution stream")
        return [
            ('CUDAExecutionProvider', {
                'device_id': str(device_id),
                'user_compute_stream': str(stream),
            }),
            'CPUExecutionProvider',
        ]


def register(registry: ConversionRegistry) -> None:
    registry.register(CudaConversionPlugin())
