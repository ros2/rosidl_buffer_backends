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
import sys
from typing import Optional
from typing import Union

import numpy as np
import onnxruntime as ort

from onnxruntime_conversions._adapter import load_external_adapters
from onnxruntime_conversions._adapter import OrtConversionRegistry
from onnxruntime_conversions._adapter import OrtTensorView
from onnxruntime_conversions._adapter import TensorMetadata
from onnxruntime_conversions._cpu_adapter import CpuOrtConversionAdapter
from tensor_msgs.msg import ExperimentalTensor


_DL_INT = 0
_DL_UINT = 1
_DL_FLOAT = 2
_DL_BFLOAT = 4
_DL_BOOL = 6

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
) -> TensorMetadata:
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
    return TensorMetadata(
        normalized_shape,
        _contiguous_strides(normalized_shape),
        normalized_type,
        dtype_code,
        dtype_bits,
        1,
        element_count,
        element_count * element_size,
        byte_offset,
        _TYPE_INFO[normalized_type][2],
    )


def _validate_message(msg: ExperimentalTensor) -> TensorMetadata:
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
    metadata: TensorMetadata,
) -> None:
    msg.dtype_code = metadata.dtype_code
    msg.dtype_bits = metadata.dtype_bits
    msg.dtype_lanes = metadata.dtype_lanes
    msg.shape = array('q', metadata.shape)
    msg.strides = array('q', metadata.strides)
    msg.byte_offset = metadata.byte_offset


def _backend_type(data: object) -> str:
    return str(getattr(data, 'backend_type', 'cpu')).lower()


_registry = OrtConversionRegistry()
_registry.register(CpuOrtConversionAdapter())
load_external_adapters(_registry)


def allocate_tensor_msg(
    shape: Sequence[int],
    element_type: object,
    device_type: str = 'auto',
    device_id: int = 0,
    stream: Optional[int] = None,
) -> ExperimentalTensor:
    metadata = _metadata_for(shape, element_type)
    msg = ExperimentalTensor()
    _set_metadata(msg, metadata)
    normalized_device = device_type.lower()
    adapter = (
        _registry.default()
        if normalized_device == 'auto'
        else _registry.for_device(normalized_device)
    )
    msg.data = adapter.allocate(metadata, device_id, stream)
    return msg


def _from_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int],
    output: bool,
) -> OrtTensorView:
    if not isinstance(msg, ExperimentalTensor):
        raise TypeError('msg must be an ExperimentalTensor')
    metadata = _validate_message(msg)
    adapter = _registry.for_data(msg.data)
    return adapter.view(msg, metadata, stream, output)


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


def _value_metadata(value: ort.OrtValue) -> TensorMetadata:
    if not isinstance(value, ort.OrtValue) or not value.is_tensor():
        raise TypeError('value must be an ONNX Runtime tensor OrtValue')
    return _metadata_for(value.shape(), value.element_type())


def to_tensor_msg(
    destination_or_value: Union[ExperimentalTensor, ort.OrtValue],
    value: Optional[ort.OrtValue] = None,
    stream: Optional[int] = None,
    device_type: str = 'auto',
) -> ExperimentalTensor:
    if value is None:
        value = destination_or_value
        metadata = _value_metadata(value)
        source_device = value.device_name().lower()
        device_id = (
            int(value.__dlpack_device__()[1])
            if source_device != 'cpu' else 0)
        destination = allocate_tensor_msg(
            metadata.shape,
            metadata.element_type,
            device_type,
            device_id,
            stream,
        )
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

    conversion_stream = (
        stream if _backend_type(destination.data) != 'cpu' else None)
    view = from_output_tensor_msg(destination, conversion_stream)
    view.value.update_inplace(value)
    view.close()
    return destination
