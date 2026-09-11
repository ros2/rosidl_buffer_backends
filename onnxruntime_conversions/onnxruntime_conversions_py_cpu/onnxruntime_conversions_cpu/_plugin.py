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

import numpy
import onnxruntime as ort

from onnxruntime_conversions._ort_bridge import make_dlpack_capsule
from onnxruntime_conversions._plugin import ConversionRegistry
from onnxruntime_conversions._plugin import TensorMetadata


class _Producer:

    def __init__(self, capsule: object, dtype: object) -> None:
        self._capsule = capsule
        self.dtype = dtype

    def __dlpack__(self, stream: Optional[int] = None, **kwargs: object) -> object:
        del stream, kwargs
        capsule = self._capsule
        if capsule is None:
            raise RuntimeError('DLPack tensor has already been consumed')
        self._capsule = None
        return capsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return (1, 0)


class CpuConversionPlugin:
    """Direct OrtValue views over host-backed tensor message storage."""

    backends = ('cpu',)
    device_types = ('cpu',)
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

    @staticmethod
    def _numpy_view(data: object, metadata: TensorMetadata) -> numpy.ndarray:
        return numpy.frombuffer(
            data,
            dtype=metadata.numpy_dtype,
            count=metadata.byte_count // metadata.numpy_dtype.itemsize,
            offset=metadata.byte_offset,
        ).reshape(tuple(metadata.shape))

    def from_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> tuple[object, object]:
        del stream
        view = self._numpy_view(data, metadata)
        dtype_code = 1 if metadata.element_type == 9 else metadata.dtype_code
        capsule = make_dlpack_capsule(
            view.ctypes.data,
            1,
            0,
            dtype_code,
            metadata.dtype_bits,
            metadata.dtype_lanes,
            list(metadata.shape),
            list(metadata.strides),
            0,
            view,
        )
        return ort.OrtValue.from_dlpack(
            _Producer(capsule, metadata.numpy_dtype)), view

    def from_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> tuple[object, object]:
        return self.from_input(data, metadata, stream)

    def copy_to(
        self,
        data: object,
        metadata: TensorMetadata,
        source: object,
        stream: Optional[int],
    ) -> None:
        del stream
        if source.device_name().lower() != 'cpu':
            raise RuntimeError('CPU plugin cannot read a non-CPU OrtValue')
        numpy.copyto(self._numpy_view(data, metadata), source.numpy())

    def session_providers(
        self, device_id: int, stream: Optional[int]
    ) -> list:
        del device_id
        if stream is not None:
            raise ValueError('Host memory sessions take no execution stream')
        return ['CPUExecutionProvider']


def register(registry: ConversionRegistry) -> None:
    registry.register(CpuConversionPlugin())
