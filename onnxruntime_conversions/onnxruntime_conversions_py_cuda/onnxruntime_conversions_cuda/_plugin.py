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

from contextlib import contextmanager
import ctypes
from pathlib import Path
from threading import Lock
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
_CUDA.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
_CUDA.cudaGetDevice.restype = ctypes.c_int
_CUDA.cudaSetDevice.argtypes = [ctypes.c_int]
_CUDA.cudaSetDevice.restype = ctypes.c_int
_CUDA.cudaGetLastError.argtypes = []
_CUDA.cudaGetLastError.restype = ctypes.c_int
_CUDA.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_int, ctypes.c_void_p,
]
_CUDA.cudaMemcpyAsync.restype = ctypes.c_int
_CUDA.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_CUDA.cudaStreamSynchronize.restype = ctypes.c_int
_CUDA.cudaStreamGetFlags.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
_CUDA.cudaStreamGetFlags.restype = ctypes.c_int

_REGISTRATION_LOCK = Lock()

_HOST_TO_DEVICE = 1
_DEVICE_TO_HOST = 2
_DEVICE_TO_DEVICE = 3
_LEGACY_DEFAULT_STREAM = 1


def _check(result: int, operation: str) -> None:
    if result != 0:
        _CUDA.cudaGetLastError()
        raise RuntimeError(f'{operation} failed with CUDA error {result}')


@contextmanager
def _device(device_id: Optional[int]):
    previous = _current_device()
    _check(_CUDA.cudaSetDevice(previous if device_id is None else device_id),
           'cudaSetDevice')
    try:
        yield
    finally:
        _check(_CUDA.cudaSetDevice(previous), 'cudaSetDevice')


def _current_device() -> int:
    device = ctypes.c_int()
    _check(_CUDA.cudaGetDevice(ctypes.byref(device)), 'cudaGetDevice')
    return device.value


def _check_device(handle: object) -> None:
    current = _current_device()
    if handle.device_id != current:
        handle.close()
        raise RuntimeError(
            f'CUDA buffer is on device {handle.device_id}, but current device is {current}')


def _stream(stream: Optional[int]) -> int:
    if stream is None:
        raise ValueError("Backend 'cuda' requires an explicit execution stream")
    return stream or _LEGACY_DEFAULT_STREAM


class _Lease:
    """Keep the CUDA mapping and backing buffer alive with the OrtValue."""

    def __init__(self, handle: object, data: object) -> None:
        self._handle = handle
        self._data = data

    def __del__(self) -> None:
        with _device(self._handle.device_id):
            self._handle.close()


class CudaConversionPlugin:
    """Direct OrtValue views over CUDA-backed tensor message storage."""

    backends = ('cuda',)
    device_types = (2,)
    priority = 100

    def create_stream(self, device_id: int) -> object:
        with _device(device_id), _REGISTRATION_LOCK:
            devices = ort.get_ep_devices()
            if not any(ep.ep_name == 'CUDAExecutionProvider' for ep in devices):
                library = Path(ort.__file__).parent / 'capi' / 'libonnxruntime_providers_cuda.so'
                ort.register_execution_provider_library('CUDAExecutionProvider', str(library))
                devices = ort.get_ep_devices()
            for ep in devices:
                if (ep.ep_name == 'CUDAExecutionProvider' and
                        int(ep.ep_options['device_id']) == device_id):
                    return ep.create_sync_stream()
        raise RuntimeError('ONNX Runtime has no stream provider for the requested CUDA device')

    def validate_stream(self, device_id: int, stream: Optional[int]) -> None:
        if not isinstance(stream, int) or isinstance(stream, bool) or stream < 0:
            raise ValueError('CUDA streams require a nonnegative integer native handle')
        with _device(device_id):
            flags = ctypes.c_uint()
            _check(_CUDA.cudaStreamGetFlags(ctypes.c_void_p(stream), ctypes.byref(flags)),
                   'cudaStreamGetFlags')

    def is_available(self) -> bool:
        count = ctypes.c_int()
        return _CUDA.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0

    def matches(self, data: object) -> bool:
        del data
        return False

    def allocate(
        self, byte_count: int, backend: str, device_id: Optional[int]
    ) -> object:
        del backend
        with _device(device_id):
            data = CudaBuffer.allocate_buffer(byte_count)
            if byte_count:
                with CudaBuffer.from_input_buffer(data, 0) as handle:
                    _check_device(handle)
            return data

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
        handle = CudaBuffer.from_input_buffer(data, _stream(stream))
        _check_device(handle)
        return self._value(handle, data, metadata)

    def from_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> tuple[object, object]:
        handle = CudaBuffer.from_output_buffer(data, _stream(stream))
        _check_device(handle)
        return self._value(handle, data, metadata)

    def copy_to(
        self,
        data: object,
        metadata: TensorMetadata,
        source: object,
        stream: Optional[int],
    ) -> None:
        cuda_stream = ctypes.c_void_p(_stream(stream))
        source_is_cpu = source.device_name().lower() == 'cpu'
        if hasattr(data, 'backend_type') and data.backend_type == 'cuda':
            with CudaBuffer.from_output_buffer(data, _stream(stream)) as handle:
                _check_device(handle)
                if not source_is_cpu and source.__dlpack_device__() != (2, handle.device_id):
                    raise ValueError(
                        'CUDA copies require matching source and destination devices')
                source_array = source.numpy() if source_is_cpu else None
                source_pointer = (
                    source_array.ctypes.data if source_is_cpu
                    else source.data_ptr()
                )
                kind = _HOST_TO_DEVICE if source_is_cpu else _DEVICE_TO_DEVICE
                with _device(handle.device_id):
                    _check(_CUDA.cudaMemcpyAsync(
                        ctypes.c_void_p(handle.device_ptr + metadata.byte_offset),
                        ctypes.c_void_p(source_pointer), metadata.byte_count,
                        kind, cuda_stream), 'cudaMemcpyAsync')
                    _check(_CUDA.cudaStreamSynchronize(cuda_stream),
                           'cudaStreamSynchronize')
            return

        if source_is_cpu:
            raise RuntimeError(
                'CUDA plugin expected a CUDA source for a CPU destination')
        destination = (ctypes.c_ubyte * len(data)).from_buffer(data)
        with _device(source.__dlpack_device__()[1]):
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
