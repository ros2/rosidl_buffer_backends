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
from collections.abc import Sequence
from dataclasses import dataclass
import sys
from types import TracebackType
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import onnxruntime as ort
from tensor_msgs.msg import ExperimentalTensor


_DL_INT = 0
_DL_UINT = 1
_DL_FLOAT = 2
_DL_BFLOAT = 4
_DL_BOOL = 6
_DL_CPU = 1
_DL_CUDA = 2

_TYPE_INFO = {
    1: (_DL_FLOAT, 32, np.dtype(np.float32)),
    2: (_DL_UINT, 8, np.dtype(np.uint8)),
    3: (_DL_INT, 8, np.dtype(np.int8)),
    4: (_DL_UINT, 16, np.dtype(np.uint16)),
    5: (_DL_INT, 16, np.dtype(np.int16)),
    6: (_DL_INT, 32, np.dtype(np.int32)),
    7: (_DL_INT, 64, np.dtype(np.int64)),
    9: (_DL_BOOL, 8, np.dtype(np.bool_)),
    10: (_DL_FLOAT, 16, np.dtype(np.float16)),
    11: (_DL_FLOAT, 64, np.dtype(np.float64)),
    12: (_DL_UINT, 32, np.dtype(np.uint32)),
    13: (_DL_UINT, 64, np.dtype(np.uint64)),
    16: (_DL_BFLOAT, 16, np.dtype(np.uint16)),
}
_NUMPY_TYPES = {
    info[2]: element_type
    for element_type, info in _TYPE_INFO.items()
    if element_type != 16
}
_DLPACK_TYPES = {
    (dtype_code, dtype_bits): element_type
    for element_type, (dtype_code, dtype_bits, _) in _TYPE_INFO.items()
}


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    element_type: int
    dtype_code: int
    dtype_bits: int
    dtype_lanes: int
    element_count: int
    byte_count: int
    byte_offset: int


class OrtTensorView:
    """Own an OrtValue and every object backing its external storage."""

    def __init__(
        self,
        value: ort.OrtValue,
        message: ExperimentalTensor,
        storage_view: object,
        backend_handle: object = None,
    ) -> None:
        self._value: Optional[ort.OrtValue] = value
        self._message: Optional[ExperimentalTensor] = message
        self._storage_view = storage_view
        self._backend_handle = backend_handle

    @property
    def value(self) -> ort.OrtValue:
        if self._value is None:
            raise RuntimeError('OrtTensorView is closed')
        return self._value

    @property
    def closed(self) -> bool:
        return self._value is None

    def close(self) -> None:
        if self.closed:
            return
        self._value = None
        self._storage_view = None
        handle = self._backend_handle
        self._backend_handle = None
        if handle is not None:
            handle.close()
        self._message = None

    def __enter__(self) -> ort.OrtValue:
        return self.value

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self.close()
        return False


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
        capsule = self._capsule
        if capsule is None:
            raise RuntimeError('DLPack tensor has already been consumed')
        self._capsule = None
        return capsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_DL_CUDA, self._device_id)


def _checked_product(values: Sequence[int], context: str) -> int:
    result = 1
    for value in values:
        if value < 0:
            raise ValueError('Tensor shape dimensions must be nonnegative')
        if value and result > sys.maxsize // value:
            raise OverflowError(context)
        result *= value
    return result


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        dimension = shape[index]
        if dimension and stride > sys.maxsize // dimension:
            raise OverflowError('Tensor strides overflow')
        stride *= dimension
    return tuple(strides)


def _normalize_element_type(element_type: object) -> int:
    if isinstance(element_type, np.dtype):
        result = _NUMPY_TYPES.get(element_type)
    elif isinstance(element_type, type):
        result = _NUMPY_TYPES.get(np.dtype(element_type))
    elif isinstance(element_type, (int, np.integer)):
        result = int(element_type)
    else:
        result = None
    if result not in _TYPE_INFO:
        raise ValueError('Unsupported ONNX tensor element type')
    return result


def _metadata_for(
    shape: Sequence[int],
    element_type: object,
    byte_offset: int = 0,
) -> _TensorMetadata:
    normalized_shape = tuple(int(dimension) for dimension in shape)
    normalized_type = _normalize_element_type(element_type)
    dtype_code, dtype_bits, _ = _TYPE_INFO[normalized_type]
    element_count = _checked_product(
        normalized_shape, 'Tensor element count overflow')
    element_size = dtype_bits // 8
    if element_count and element_count > sys.maxsize // element_size:
        raise OverflowError('Tensor byte count overflow')
    if byte_offset < 0:
        raise ValueError('Tensor byte_offset must be nonnegative')
    if byte_offset % element_size:
        raise ValueError('Tensor byte_offset is not element-aligned')
    return _TensorMetadata(
        normalized_shape,
        _contiguous_strides(normalized_shape),
        normalized_type,
        dtype_code,
        dtype_bits,
        1,
        element_count,
        element_count * element_size,
        byte_offset,
    )


def _validate_message(msg: ExperimentalTensor) -> _TensorMetadata:
    if int(msg.dtype_lanes) != 1:
        raise ValueError('ONNX Runtime tensors require dtype_lanes == 1')
    element_type = _DLPACK_TYPES.get(
        (int(msg.dtype_code), int(msg.dtype_bits)))
    if element_type is None:
        raise ValueError(
            'ExperimentalTensor dtype is unsupported by ONNX Runtime')
    metadata = _metadata_for(msg.shape, element_type, int(msg.byte_offset))
    strides = tuple(int(stride) for stride in msg.strides)
    if strides and strides != metadata.strides:
        raise ValueError(
            'ONNX Runtime conversion requires contiguous tensor strides')
    if metadata.byte_offset > len(msg.data):
        raise ValueError('Tensor byte_offset exceeds its backing buffer')
    if metadata.byte_count > len(msg.data) - metadata.byte_offset:
        raise ValueError('Tensor view exceeds its backing buffer')
    return metadata


def _set_metadata(
    msg: ExperimentalTensor,
    metadata: _TensorMetadata,
) -> None:
    msg.dtype_code = metadata.dtype_code
    msg.dtype_bits = metadata.dtype_bits
    msg.dtype_lanes = metadata.dtype_lanes
    msg.shape = array('q', metadata.shape)
    msg.strides = array('q', metadata.strides)
    msg.byte_offset = metadata.byte_offset


def _backend_type(data: object) -> str:
    return str(getattr(data, 'backend_type', 'cpu')).lower()


def _require_cuda() -> tuple[object, object]:
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError(
            'CUDA tensor conversion requires CUDAExecutionProvider')
    from cuda_buffer import CudaBuffer
    from onnxruntime_conversions import _dlpack_bridge
    return CudaBuffer, _dlpack_bridge


def allocate_tensor_msg(
    shape: Sequence[int],
    element_type: object,
    device_type: str = 'cpu',
    device_id: int = 0,
) -> ExperimentalTensor:
    metadata = _metadata_for(shape, element_type)
    msg = ExperimentalTensor()
    _set_metadata(msg, metadata)
    normalized_device = device_type.lower()
    if normalized_device == 'cpu':
        if device_id != 0:
            raise ValueError('CPU tensors require device_id 0')
        msg.data = array('B', bytes(metadata.byte_count))
    elif normalized_device == 'cuda':
        if device_id != 0:
            raise ValueError(
                'cuda_buffer currently supports allocation on device 0 only')
        if metadata.byte_count == 0:
            raise ValueError('CUDA zero-sized tensors are not supported')
        cuda_buffer, _ = _require_cuda()
        msg.data = cuda_buffer.allocate_buffer(metadata.byte_count)
    else:
        raise ValueError(f'Unsupported tensor device type: {device_type}')
    return msg


def _cpu_view(
    msg: ExperimentalTensor,
    metadata: _TensorMetadata,
) -> OrtTensorView:
    dtype = _TYPE_INFO[metadata.element_type][2]
    storage = np.ndarray(
        metadata.shape,
        dtype=dtype,
        buffer=msg.data,
        offset=metadata.byte_offset,
        order='C',
    )
    if metadata.element_type == 16:
        value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
            storage, metadata.element_type)
    else:
        value = ort.OrtValue.ortvalue_from_numpy(storage)
    return OrtTensorView(value, msg, storage)


def _cuda_view(
    msg: ExperimentalTensor,
    metadata: _TensorMetadata,
    stream: Optional[int],
    output: bool,
) -> OrtTensorView:
    cuda_buffer, bridge = _require_cuda()
    resolved_stream = (
        cuda_buffer.get_internal_stream() if stream is None else int(stream))
    if resolved_stream == 0:
        raise RuntimeError('CUDA tensor conversion requires a valid stream')
    factory = (
        cuda_buffer.from_output_buffer if output
        else cuda_buffer.from_input_buffer)
    handle = factory(msg.data, resolved_stream)
    capsule_code = (
        _DL_UINT if metadata.element_type == 9 else metadata.dtype_code)
    capsule = bridge.make_dlpack_capsule(
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
    return OrtTensorView(value, msg, producer, handle)


def _from_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int],
    output: bool,
) -> OrtTensorView:
    if not isinstance(msg, ExperimentalTensor):
        raise TypeError('msg must be an ExperimentalTensor')
    metadata = _validate_message(msg)
    backend = _backend_type(msg.data)
    if backend == 'cpu':
        if stream is not None:
            raise ValueError('CPU tensor conversion does not accept a stream')
        return _cpu_view(msg, metadata)
    if backend == 'cuda':
        return _cuda_view(msg, metadata, stream, output)
    raise ValueError(f'Unsupported tensor buffer backend: {backend}')


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
) -> OrtTensorView:
    return _from_tensor_msg(msg, stream, False)


def from_output_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
) -> OrtTensorView:
    return _from_tensor_msg(msg, stream, True)


def _value_metadata(value: ort.OrtValue) -> _TensorMetadata:
    if not isinstance(value, ort.OrtValue) or not value.is_tensor():
        raise TypeError('value must be an ONNX Runtime tensor OrtValue')
    return _metadata_for(value.shape(), value.element_type())


def to_tensor_msg(
    destination_or_value: Union[ExperimentalTensor, ort.OrtValue],
    value: Optional[ort.OrtValue] = None,
    stream: Optional[int] = None,
) -> ExperimentalTensor:
    if value is None:
        value = destination_or_value
        metadata = _value_metadata(value)
        device = value.device_name().lower()
        device_id = (
            int(value.__dlpack_device__()[1]) if device == 'cuda' else 0)
        destination = allocate_tensor_msg(
            metadata.shape, metadata.element_type, device, device_id)
    else:
        if not isinstance(destination_or_value, ExperimentalTensor):
            raise TypeError('destination must be an ExperimentalTensor')
        destination = destination_or_value
        metadata = _value_metadata(value)
        if metadata.byte_count > len(destination.data):
            raise ValueError('OrtValue tensor exceeds the destination buffer')
        metadata = _metadata_for(
            metadata.shape, metadata.element_type, int(destination.byte_offset))
        if metadata.byte_count > len(destination.data) - metadata.byte_offset:
            raise ValueError('OrtValue tensor exceeds the destination view')
        _set_metadata(destination, metadata)

    view = from_output_tensor_msg(destination, stream)
    view.value.update_inplace(value)
    view.close()
    return destination
