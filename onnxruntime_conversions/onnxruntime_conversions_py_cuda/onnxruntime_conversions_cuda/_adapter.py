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

"""Registered CUDA adapter for Python ONNX Runtime conversions."""

from importlib.util import find_spec
from typing import Optional

import numpy as np
import onnxruntime as ort

from onnxruntime_conversions import _dlpack_bridge
from onnxruntime_conversions._adapter import OrtConversionRegistry
from onnxruntime_conversions._adapter import OrtTensorView
from onnxruntime_conversions._adapter import TensorMetadata
from tensor_msgs.msg import ExperimentalTensor


_DL_UINT = 1
_DL_CUDA = 2


def _cuda_buffer_installed() -> bool:
    return find_spec('cuda_buffer') is not None


def _cuda_buffer():
    from cuda_buffer import CudaBuffer
    return CudaBuffer


def _require_stream(stream: Optional[int]) -> int:
    if not isinstance(stream, int) or isinstance(stream, bool):
        raise TypeError(
            'CUDA tensor conversion requires an explicit integer stream')
    if stream <= 0:
        raise ValueError(
            'CUDA tensor conversion requires a positive nonzero explicit '
            'integer stream')
    return stream


class _DLPackProducer:
    def __init__(
        self,
        capsule: object,
        device_id: int,
        is_bool: bool,
    ) -> None:
        self._capsule = capsule
        self._device_id = device_id
        if is_bool:
            self.dtype = np.dtype(np.bool_)

    def __dlpack__(self, stream: Optional[int] = None) -> object:
        del stream
        capsule = self._capsule
        if capsule is None:
            raise RuntimeError('DLPack tensor has already been consumed')
        self._capsule = None
        return capsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_DL_CUDA, self._device_id)


class CudaOrtConversionAdapter:
    """Create ONNX Runtime views over CUDA message storage."""

    device_type = 'cuda'
    buffer_backend = 'cuda'
    priority = 100

    def is_available(self) -> bool:
        return (
            _cuda_buffer_installed() and
            'CUDAExecutionProvider' in ort.get_available_providers()
        )

    def unavailable_error(self) -> RuntimeError:
        if not _cuda_buffer_installed():
            return RuntimeError(
                'CUDA conversion requires the cuda_buffer_py package')
        return RuntimeError(
            'CUDA tensor conversion requires CUDAExecutionProvider')

    def allocate(
        self,
        metadata: TensorMetadata,
        device_id: int,
        stream: Optional[int],
    ) -> object:
        _require_stream(stream)
        if device_id != 0:
            raise ValueError(
                'cuda_buffer currently supports allocation on device 0 only')
        if metadata.byte_count == 0:
            raise ValueError('CUDA zero-sized tensors are not supported')
        return _cuda_buffer().allocate_buffer(metadata.byte_count)

    def view(
        self,
        message: ExperimentalTensor,
        metadata: TensorMetadata,
        stream: Optional[int],
        output: bool,
    ) -> OrtTensorView:
        cuda_buffer = _cuda_buffer()
        cuda_stream = _require_stream(stream)
        factory = (
            cuda_buffer.from_output_buffer
            if output else cuda_buffer.from_input_buffer
        )
        handle = factory(message.data, cuda_stream)
        capsule_code = (
            _DL_UINT if metadata.element_type == 9 else metadata.dtype_code)
        capsule = _dlpack_bridge.make_dlpack_capsule(
            handle.device_ptr,
            _DL_CUDA,
            handle.device_id,
            capsule_code,
            metadata.dtype_bits,
            metadata.dtype_lanes,
            metadata.shape,
            metadata.byte_offset,
            handle,
        )
        producer = _DLPackProducer(
            capsule, handle.device_id, metadata.element_type == 9)
        value = ort.OrtValue.from_dlpack(producer)
        return OrtTensorView(value, message, producer, handle)


def register(registry: OrtConversionRegistry) -> None:
    """Register the CUDA conversion adapter."""
    registry.register(CudaOrtConversionAdapter())
