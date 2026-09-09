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

from torch_conversions._plugin import load_external_plugins
from torch_conversions._plugin import TensorMetadata
from torch_conversions._plugin import TorchConversionRegistry


_DTYPE_TO_DLPACK = {
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
_DLPACK_TO_DTYPE = {value: key for key, value in _DTYPE_TO_DLPACK.items()}

_registry = TorchConversionRegistry()
load_external_plugins(_registry)


def _contiguous_strides(shape: Sequence[int]) -> list[int]:
    strides = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        stride *= shape[index]
    return strides


def _metadata(msg: ExperimentalTensor) -> TensorMetadata:
    shape = list(msg.shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError('Tensor shape dimensions must be nonnegative')
    strides = list(msg.strides) or _contiguous_strides(shape)
    if len(strides) != len(shape) or any(stride < 0 for stride in strides):
        raise ValueError('Tensor strides must match rank and be nonnegative')
    key = (msg.dtype_code, msg.dtype_bits, msg.dtype_lanes)
    if key not in _DLPACK_TO_DTYPE:
        raise TypeError(f'Unsupported DLPack dtype {key}')
    dtype = _DLPACK_TO_DTYPE[key]
    element_size = torch.empty((), dtype=dtype).element_size()
    span = 0 if 0 in shape else 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(shape, strides)
    )
    required_size = msg.byte_offset + span * element_size
    if required_size > len(msg.data):
        raise ValueError(
            f'Tensor view needs {required_size} bytes; buffer has {len(msg.data)}'
        )
    return TensorMetadata(
        shape, strides, dtype, span, msg.byte_offset,
        msg.dtype_code, msg.dtype_bits, msg.dtype_lanes,
    )


def _plugin_available(device: Union[str, torch.device]) -> bool:
    try:
        _registry.for_device(torch.device(device))
    except (RuntimeError, ValueError):
        return False
    return True


def allocate_tensor_msg(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]] = None,
) -> ExperimentalTensor:
    normalized_shape = list(shape)
    if any(dimension < 0 for dimension in normalized_shape):
        raise ValueError('Tensor shape dimensions must be nonnegative')
    if dtype not in _DTYPE_TO_DLPACK:
        raise TypeError(f'Unsupported torch dtype {dtype}')
    selected = (
        _registry.default_device() if device is None else torch.device(device)
    )
    plugin = _registry.for_device(selected)
    msg = ExperimentalTensor()
    code, bits, lanes = _DTYPE_TO_DLPACK[dtype]
    msg.dtype_code = code
    msg.dtype_bits = bits
    msg.dtype_lanes = lanes
    msg.shape = normalized_shape
    msg.strides = _contiguous_strides(normalized_shape)
    msg.byte_offset = 0
    msg.data = plugin.allocate(prod(normalized_shape) * bits * lanes // 8, selected)
    return msg


def from_output_tensor_msg(
    msg: ExperimentalTensor,
) -> Optional[torch.Tensor]:
    if len(msg.data) == 0:
        return None
    return _registry.for_data(msg.data).from_output(msg.data, _metadata(msg))


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    clone: bool = True,
) -> Optional[torch.Tensor]:
    if len(msg.data) == 0:
        return None
    tensor = _registry.for_data(msg.data).from_input(msg.data, _metadata(msg))
    return tensor.clone() if clone else tensor


def to_tensor_msg(*args: object) -> ExperimentalTensor:
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
            'Expected to_tensor_msg(tensor) or to_tensor_msg(msg, tensor)'
        )
    contiguous = tensor.contiguous()
    required_size = contiguous.numel() * contiguous.element_size()
    if required_size > len(msg.data):
        raise ValueError('Tensor exceeds allocated message storage')
    code, bits, lanes = _DTYPE_TO_DLPACK[contiguous.dtype]
    msg.dtype_code, msg.dtype_bits, msg.dtype_lanes = code, bits, lanes
    msg.shape = list(contiguous.shape)
    msg.strides = _contiguous_strides(msg.shape)
    msg.byte_offset = 0
    output = from_output_tensor_msg(msg)
    if output is not None:
        output.copy_(contiguous)
    return msg


def set_stream(
    device: Optional[Union[str, torch.device]] = None,
):
    selected = (
        _registry.default_device() if device is None else torch.device(device)
    )
    return _registry.for_device(selected).stream_context()
