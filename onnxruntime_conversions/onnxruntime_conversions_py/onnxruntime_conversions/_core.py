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

from math import prod
from types import TracebackType
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

import numpy
import onnxruntime as ort

from onnxruntime_conversions._plugin import ConversionRegistry
from onnxruntime_conversions._plugin import load_external_plugins
from onnxruntime_conversions._plugin import TensorMetadata
from tensor_msgs.msg import ExperimentalTensor


# ONNX tensor element type -> (message dtype code, bits, numpy dtype).
_ELEMENT_TYPES = {
    1: (2, 32, numpy.dtype(numpy.float32)),
    2: (1, 8, numpy.dtype(numpy.uint8)),
    3: (0, 8, numpy.dtype(numpy.int8)),
    4: (1, 16, numpy.dtype(numpy.uint16)),
    5: (0, 16, numpy.dtype(numpy.int16)),
    6: (0, 32, numpy.dtype(numpy.int32)),
    7: (0, 64, numpy.dtype(numpy.int64)),
    9: (6, 8, numpy.dtype(numpy.bool_)),
    10: (2, 16, numpy.dtype(numpy.float16)),
    11: (2, 64, numpy.dtype(numpy.float64)),
    12: (1, 32, numpy.dtype(numpy.uint32)),
    13: (1, 64, numpy.dtype(numpy.uint64)),
    16: (4, 16, numpy.dtype(numpy.uint16)),
}
_FROM_NUMPY = {
    info[2]: element_type
    for element_type, info in _ELEMENT_TYPES.items()
    if element_type != 16
}
_FROM_MESSAGE = {
    (code, bits): element_type
    for element_type, (code, bits, _) in _ELEMENT_TYPES.items()
}

ElementType = Union[int, numpy.dtype, type]
_REGISTRY = ConversionRegistry()
load_external_plugins(_REGISTRY)


def _element_type(element_type: ElementType) -> int:
    if isinstance(element_type, numpy.dtype):
        resolved = _FROM_NUMPY.get(element_type)
    elif isinstance(element_type, type):
        resolved = _FROM_NUMPY.get(numpy.dtype(element_type))
    elif isinstance(element_type, (int, numpy.integer)):
        resolved = int(element_type)
    else:
        resolved = None
    if resolved not in _ELEMENT_TYPES:
        raise ValueError(f'Unsupported ONNX tensor element type: {element_type!r}')
    return resolved


def contiguous_strides(shape: Sequence[int]) -> list[int]:
    result = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        result[index] = stride
        stride *= shape[index]
    return result


def _metadata(msg: ExperimentalTensor) -> TensorMetadata:
    if int(msg.dtype_lanes) != 1:
        raise ValueError('ONNX Runtime tensors require dtype_lanes == 1')
    element_type = _FROM_MESSAGE.get(
        (int(msg.dtype_code), int(msg.dtype_bits)))
    if element_type is None:
        raise ValueError(
            'ExperimentalTensor dtype is unsupported by ONNX Runtime')
    shape = list(msg.shape)
    strides = list(msg.strides) or contiguous_strides(shape)
    if len(shape) != len(strides):
        raise ValueError('Strides must match shape rank and be contiguous')
    if any(value < 0 for value in shape + strides):
        raise ValueError('Shape dimensions and strides must be nonnegative')
    if strides != contiguous_strides(shape):
        raise ValueError(
            'ONNX Runtime conversion requires contiguous tensor strides')
    numpy_dtype = _ELEMENT_TYPES[element_type][2]
    byte_count = prod(shape) * numpy_dtype.itemsize
    if msg.byte_offset > len(msg.data) or byte_count > (
            len(msg.data) - msg.byte_offset):
        raise ValueError(
            f'Tensor view needs {byte_count} bytes but buffer has '
            f'{len(msg.data) - min(msg.byte_offset, len(msg.data))}')
    return TensorMetadata(
        shape, strides, element_type, numpy_dtype,
        int(msg.dtype_code), int(msg.dtype_bits), int(msg.dtype_lanes),
        byte_count, int(msg.byte_offset),
    )


class OrtTensorView:
    """Keep an OrtValue and its message-storage lease alive together."""

    def __init__(self, value: object, lease: object = None) -> None:
        self._value: Optional[object] = value
        self._lease: Optional[object] = lease

    @property
    def value(self) -> object:
        if self._value is None:
            raise RuntimeError('OrtTensorView is closed')
        return self._value

    @property
    def closed(self) -> bool:
        return self._value is None

    def close(self) -> None:
        self._value = None
        self._lease = None

    def __enter__(self) -> object:
        return self.value

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self.close()
        return False


def available_backends() -> list[str]:
    return _REGISTRY.backends()


def default_backend() -> str:
    return _REGISTRY.default_backend()


def allocate_tensor_msg(
    shape: Sequence[int],
    element_type: ElementType,
    backend: Optional[str] = None,
) -> ExperimentalTensor:
    resolved = _element_type(element_type)
    code, bits, numpy_dtype = _ELEMENT_TYPES[resolved]
    dimensions = list(shape)
    if any(value < 0 for value in dimensions):
        raise ValueError('Shape dimensions must be nonnegative')
    plugin = _REGISTRY.for_backend(backend or _REGISTRY.default_backend())
    msg = ExperimentalTensor()
    msg.shape = dimensions
    msg.strides = contiguous_strides(dimensions)
    msg.dtype_code, msg.dtype_bits, msg.dtype_lanes = code, bits, 1
    msg.byte_offset = 0
    msg.data = plugin.allocate(prod(dimensions) * numpy_dtype.itemsize,
                               plugin.backends[0])
    return msg


def _view(
    msg: ExperimentalTensor,
    stream: Optional[int],
    output: bool,
) -> Optional[OrtTensorView]:
    if len(msg.data) == 0:
        return None
    plugin = _REGISTRY.for_data(msg.data)
    conversion = plugin.from_output if output else plugin.from_input
    value, lease = conversion(msg.data, _metadata(msg), stream)
    return OrtTensorView(value, lease)


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
) -> Optional[OrtTensorView]:
    return _view(msg, stream, False)


def from_output_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
) -> Optional[OrtTensorView]:
    return _view(msg, stream, True)


def to_tensor_msg(
    destination_or_value: Union[ExperimentalTensor, ort.OrtValue],
    value: Optional[ort.OrtValue] = None,
    stream: Optional[int] = None,
    backend: Optional[str] = None,
) -> ExperimentalTensor:
    if value is None:
        value = destination_or_value
        destination = None
    else:
        destination = destination_or_value
    if not isinstance(value, ort.OrtValue) or not value.is_tensor():
        raise TypeError('value must be an ONNX Runtime tensor OrtValue')

    shape = list(value.shape())
    element_type = int(value.element_type())
    _, _, numpy_dtype = _ELEMENT_TYPES[_element_type(element_type)]
    byte_count = prod(shape) * numpy_dtype.itemsize
    if destination is None:
        selected = backend
        if selected is None:
            selected = _REGISTRY.for_device(
                value.device_name().lower()).backends[0]
        destination = allocate_tensor_msg(shape, element_type, selected)
    if not isinstance(destination, ExperimentalTensor):
        raise TypeError('destination must be an ExperimentalTensor')
    if byte_count > len(destination.data):
        raise ValueError(
            f'OrtValue needs {byte_count} bytes but destination buffer has '
            f'{len(destination.data)}')

    code, bits, _ = _ELEMENT_TYPES[_element_type(element_type)]
    destination.shape = shape
    destination.strides = contiguous_strides(shape)
    destination.dtype_code = code
    destination.dtype_bits = bits
    destination.dtype_lanes = 1
    destination.byte_offset = 0
    metadata = _metadata(destination)
    source_device = value.device_name().lower()
    plugin = _REGISTRY.for_data(destination.data) if source_device == 'cpu' \
        else _REGISTRY.for_device(source_device)
    plugin.copy_to(destination.data, metadata, value, stream)
    return destination


def session_providers(
    backend: Optional[str] = None,
    device_id: int = 0,
    stream: Optional[int] = None,
) -> list:
    selected = backend or _REGISTRY.default_backend()
    if selected not in _REGISTRY.backends():
        if selected != 'cpu' and stream is None:
            raise ValueError(
                f'Backend {selected!r} requires an explicit execution stream')
        raise ValueError(
            f'No ONNX Runtime execution provider is installed for backend '
            f'{selected!r}; install its conversion plugin')
    plugin = _REGISTRY.for_backend(selected)
    return plugin.session_providers(device_id, stream)
