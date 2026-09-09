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

from types import TracebackType
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

import dlpack_conversions
from dlpack_conversions._dlpack_bridge import capsule_device

import numpy

import onnxruntime as ort

from tensor_msgs.msg import ExperimentalTensor


# ONNX tensor element type -> (DLPack dtype code, bits, numpy dtype).
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
_BOOL = 9
# bfloat16 shares numpy's uint16, so it cannot be reached from a numpy dtype.
_FROM_NUMPY = {
    info[2]: element_type
    for element_type, info in _ELEMENT_TYPES.items()
    if element_type != 16
}
_FROM_DLPACK = {
    (code, bits): element_type
    for element_type, (code, bits, _) in _ELEMENT_TYPES.items()
}

ElementType = Union[int, numpy.dtype, type]


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


def _element_type_of(msg: ExperimentalTensor) -> int:
    if int(msg.dtype_lanes) != 1:
        raise ValueError('ONNX Runtime tensors require dtype_lanes == 1')
    element_type = _FROM_DLPACK.get((int(msg.dtype_code), int(msg.dtype_bits)))
    if element_type is None:
        raise ValueError(
            'ExperimentalTensor dtype is unsupported by ONNX Runtime')
    return element_type


class _Producer:
    """Presents one DLPack capsule through the array API protocol."""

    def __init__(self, capsule: object, dtype: numpy.dtype):
        self._capsule = capsule
        self._device = capsule_device(capsule)
        self.dtype = dtype

    def __dlpack__(self, stream: Optional[int] = None, **kwargs: object) -> object:
        del stream, kwargs
        capsule = self._capsule
        if capsule is None:
            raise RuntimeError('DLPack tensor has already been consumed')
        self._capsule = None
        return capsule

    def __dlpack_device__(self) -> tuple:
        return self._device


class OrtTensorView:
    """An OrtValue over tensor message storage.

    ONNX Runtime takes over the DLPack deleter, so the value itself holds the
    storage lease. Keep the view alive while ONNX Runtime reads or writes it.
    """

    def __init__(self, value: ort.OrtValue) -> None:
        self._value: Optional[ort.OrtValue] = value

    @property
    def value(self) -> ort.OrtValue:
        if self._value is None:
            raise RuntimeError('OrtTensorView is closed')
        return self._value

    @property
    def closed(self) -> bool:
        return self._value is None

    def close(self) -> None:
        self._value = None

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


def available_backends() -> list:
    return dlpack_conversions.available_backends()


def default_backend() -> str:
    return dlpack_conversions.default_backend()


def allocate_tensor_msg(
    shape: Sequence[int],
    element_type: ElementType,
    backend: Optional[str] = None,
) -> ExperimentalTensor:
    code, bits, _ = _ELEMENT_TYPES[_element_type(element_type)]
    return dlpack_conversions.allocate_tensor_msg(shape, (code, bits, 1), backend)


def _prepare(msg: ExperimentalTensor) -> tuple:
    element_type = _element_type_of(msg)
    strides = list(msg.strides)
    if strides and strides != dlpack_conversions.contiguous_strides(
            list(msg.shape)):
        raise ValueError(
            'ONNX Runtime conversion requires contiguous tensor strides')
    # ONNX Runtime rejects the DLPack bool code and instead infers bool from
    # the producer dtype, so bool storage is described as the uint8 it is.
    dtype = (1, 8, 1) if element_type == _BOOL else None
    return element_type, dtype


def _view(
    element_type: int,
    capsule: Optional[object],
) -> Optional[OrtTensorView]:
    if capsule is None:
        return None
    producer = _Producer(capsule, _ELEMENT_TYPES[element_type][2])
    return OrtTensorView(ort.OrtValue.from_dlpack(producer))


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
) -> Optional[OrtTensorView]:
    element_type, dtype = _prepare(msg)
    return _view(
        element_type,
        dlpack_conversions.from_input_tensor_msg(msg, stream, dtype),
    )


def from_output_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
) -> Optional[OrtTensorView]:
    element_type, dtype = _prepare(msg)
    return _view(
        element_type,
        dlpack_conversions.from_output_tensor_msg(msg, stream, dtype),
    )


def to_tensor_msg(
    destination_or_value: Union[ExperimentalTensor, ort.OrtValue],
    value: Optional[ort.OrtValue] = None,
    stream: Optional[int] = None,
    backend: Optional[str] = None,
) -> ExperimentalTensor:
    """Copy an OrtValue into message storage and stamp its metadata.

    Called with one argument, allocates a message on the backend that owns the
    value. Called with two, copies into the message given first. The copy runs
    through the storage plugin, so device values never stage through the host.
    """
    if value is None:
        value = destination_or_value
        destination = None
    else:
        destination = destination_or_value

    if not isinstance(value, ort.OrtValue) or not value.is_tensor():
        raise TypeError('value must be an ONNX Runtime tensor OrtValue')

    capsule = value.__dlpack__()
    if destination is None:
        copied = dlpack_conversions.to_tensor_msg(
            capsule, stream=stream, backend=backend)
    else:
        copied = dlpack_conversions.to_tensor_msg(
            destination, capsule, stream)

    # ONNX Runtime exports bool storage as uint8, so restore the element type
    # the value actually carries.
    if value.element_type() == _BOOL:
        copied.dtype_code, copied.dtype_bits, _ = _ELEMENT_TYPES[_BOOL]
    return copied


def session_providers(
    backend: Optional[str] = None,
    device_id: int = 0,
    stream: Optional[int] = None,
) -> list:
    """Provider list that runs a session where the given backend allocates.

    Only the backends ONNX Runtime ships a provider for are handled. For any
    other backend, build the provider list yourself.
    """
    selected = backend or dlpack_conversions.default_backend()
    if selected == 'cpu':
        if stream is not None:
            raise ValueError('Host memory sessions take no execution stream')
        return ['CPUExecutionProvider']
    if stream is None:
        raise ValueError(
            f'Backend {selected!r} requires an explicit execution stream')
    if selected == 'cuda':
        provider = 'CUDAExecutionProvider'
    elif selected == 'rocm':
        provider = 'ROCMExecutionProvider'
    else:
        raise ValueError(
            f'No ONNX Runtime execution provider is known for backend '
            f'{selected!r}; build the provider list yourself'
        )
    return [
        (provider, {
            'device_id': str(device_id),
            'user_compute_stream': str(stream),
        }),
        'CPUExecutionProvider',
    ]
