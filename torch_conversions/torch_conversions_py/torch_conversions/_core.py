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
from typing import Optional
from typing import Sequence
from typing import Union

from tensor_msgs.msg import ExperimentalTensor

import torch

from torch_conversions._plugin import ConversionRegistry
from torch_conversions._plugin import load_external_plugins
from torch_conversions._plugin import TensorMetadata


_DTYPE_TO_MESSAGE = {
    torch.uint8: (1, 8, 1),
    torch.int8: (0, 8, 1),
    torch.int16: (0, 16, 1),
    torch.int32: (0, 32, 1),
    torch.int64: (0, 64, 1),
    torch.float16: (2, 16, 1),
    torch.bfloat16: (4, 16, 1),
    torch.float32: (2, 32, 1),
    torch.float64: (2, 64, 1),
    torch.bool: (6, 8, 1),
}
_MESSAGE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_MESSAGE.items()}

Device = Union[str, torch.device]
_REGISTRY = ConversionRegistry()
load_external_plugins(_REGISTRY)


def contiguous_strides(shape: Sequence[int]) -> list[int]:
    result = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        result[index] = stride
        stride *= shape[index]
    return result


def _metadata(msg: ExperimentalTensor) -> TensorMetadata:
    key = (msg.dtype_code, msg.dtype_bits, msg.dtype_lanes)
    dtype = _MESSAGE_TO_DTYPE.get(key)
    if dtype is None:
        raise TypeError(f'Unsupported tensor message dtype {key}')
    shape = list(msg.shape)
    strides = list(msg.strides) or contiguous_strides(shape)
    if len(shape) != len(strides):
        raise ValueError('Strides must match shape rank')
    if any(value < 0 for value in shape + strides):
        raise ValueError('Negative shape dimensions and strides are invalid')
    span = 0 if any(value == 0 for value in shape) else 1
    if span:
        span += sum((size - 1) * stride for size, stride in zip(shape, strides))
    item_size = torch.empty((), dtype=dtype).element_size()
    if msg.byte_offset + span * item_size > len(msg.data):
        raise ValueError('Tensor view exceeds message storage')
    return TensorMetadata(
        shape, strides, dtype, *key, span, msg.byte_offset, item_size)


def available_backends() -> list[str]:
    return _REGISTRY.backends()


def backend_available(backend: str) -> bool:
    return backend in _REGISTRY.backends()


def backend_for_device(device: Device) -> Optional[str]:
    device_type = torch.device(device).type
    return _REGISTRY.backend_for_device(device_type)


def default_backend() -> str:
    return _REGISTRY.default_backend()


def allocate_tensor_msg(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: Optional[Device] = None,
) -> ExperimentalTensor:
    dtype_description = _DTYPE_TO_MESSAGE.get(dtype)
    if dtype_description is None:
        raise TypeError(f'Unsupported torch dtype {dtype}')
    dimensions = list(shape)
    if any(value < 0 for value in dimensions):
        raise ValueError('Shape dimensions must be nonnegative')
    if device is None:
        plugin = _REGISTRY.for_backend(_REGISTRY.default_backend())
        selected_device = torch.device(plugin.device_types[0])
    else:
        selected_device = torch.device(device)
        plugin = _REGISTRY.for_device(selected_device.type)
    backend = plugin.backends[0]
    msg = ExperimentalTensor()
    msg.shape = dimensions
    msg.strides = contiguous_strides(dimensions)
    msg.dtype_code, msg.dtype_bits, msg.dtype_lanes = dtype_description
    msg.byte_offset = 0
    byte_count = prod(dimensions) * torch.empty((), dtype=dtype).element_size()
    msg.data = plugin.allocate(byte_count, backend, selected_device)
    return msg


def from_output_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if len(msg.data) == 0:
        return None
    return _REGISTRY.for_data(msg.data).from_output(
        msg.data, _metadata(msg), stream)


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    clone: bool = True,
    stream: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if len(msg.data) == 0:
        return None
    return _REGISTRY.for_data(msg.data).from_input(
        msg.data, _metadata(msg), stream, clone)


def to_tensor_msg(
    *args: object,
    stream: Optional[int] = None,
) -> ExperimentalTensor:
    if len(args) == 1 and isinstance(args[0], torch.Tensor):
        tensor = args[0]
        if tensor.numel() == 0:
            return ExperimentalTensor()
        msg = allocate_tensor_msg(tensor.shape, tensor.dtype, tensor.device)
    elif (
        len(args) == 2
        and isinstance(args[0], ExperimentalTensor)
        and isinstance(args[1], torch.Tensor)
    ):
        msg, tensor = args
        if tensor.numel() == 0:
            return msg
    else:
        raise TypeError(
            'Expected to_tensor_msg(tensor) or to_tensor_msg(msg, tensor)')

    dtype_description = _DTYPE_TO_MESSAGE.get(tensor.dtype)
    if dtype_description is None:
        raise TypeError(f'Unsupported torch dtype {tensor.dtype}')
    if tensor.numel() * tensor.element_size() > len(msg.data):
        raise ValueError('Tensor exceeds allocated message storage')

    msg.shape = list(tensor.shape)
    msg.strides = contiguous_strides(msg.shape)
    msg.dtype_code, msg.dtype_bits, msg.dtype_lanes = dtype_description
    msg.byte_offset = 0
    plugin = _REGISTRY.for_data(msg.data) if tensor.device.type == 'cpu' \
        else _REGISTRY.for_device(tensor.device.type)
    plugin.copy_to(
        msg.data, _metadata(msg), tensor, stream)
    return msg


def set_stream(device: Optional[Device] = None):
    plugin = _REGISTRY.for_backend(_REGISTRY.default_backend()) if device is None \
        else _REGISTRY.for_device(torch.device(device).type)
    selected = torch.device(device or plugin.device_types[0])
    return plugin.stream_context(selected)
