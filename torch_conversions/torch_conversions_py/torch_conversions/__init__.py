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

"""Convert ExperimentalTensor messages to and from PyTorch tensors."""

from math import prod
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Union

from tensor_msgs.msg import ExperimentalTensor

import torch

from torch_conversions._adapter import TensorMetadata
from torch_conversions._adapter import TorchConversionRegistry
from torch_conversions._cpu_adapter import CpuTorchConversionAdapter
from torch_conversions._cuda_adapter import CudaTorchConversionAdapter


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
_registry.register(CpuTorchConversionAdapter())
_registry.register(CudaTorchConversionAdapter())


def _contiguous_strides(shape: Sequence[int]) -> list[int]:
    strides = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        stride *= shape[index]
    return strides


def _dtype_from_msg(msg: ExperimentalTensor) -> torch.dtype:
    key = (msg.dtype_code, msg.dtype_bits, msg.dtype_lanes)
    try:
        return _DLPACK_TO_DTYPE[key]
    except KeyError as error:
        raise TypeError(f'Unsupported DLPack dtype {key}') from error


def _set_metadata(msg: ExperimentalTensor, tensor: torch.Tensor) -> None:
    try:
        dtype_code, dtype_bits, dtype_lanes = _DTYPE_TO_DLPACK[tensor.dtype]
    except KeyError as error:
        raise TypeError(f'Unsupported torch dtype {tensor.dtype}') from error
    msg.dtype_code = dtype_code
    msg.dtype_bits = dtype_bits
    msg.dtype_lanes = dtype_lanes
    msg.shape = list(tensor.shape)
    msg.strides = _contiguous_strides(msg.shape)
    msg.byte_offset = 0


def _metadata(
    msg: ExperimentalTensor,
) -> TensorMetadata:
    shape = list(msg.shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError('Tensor shape dimensions must be nonnegative')
    strides = list(msg.strides) or _contiguous_strides(shape)
    if len(strides) != len(shape):
        raise ValueError(
            'Tensor strides must be empty or match the shape rank'
        )
    if any(stride < 0 for stride in strides):
        raise ValueError('Negative tensor strides are unsupported')

    dtype = _dtype_from_msg(msg)
    element_size = torch.empty((), dtype=dtype).element_size()
    span = 0 if 0 in shape else 1
    if span:
        span += sum(
            (dimension - 1) * stride
            for dimension, stride in zip(shape, strides)
        )
    required_size = msg.byte_offset + span * element_size
    if required_size > len(msg.data):
        raise ValueError(
            f'Tensor view requires {required_size} bytes, '
            f'but the buffer has {len(msg.data)}'
        )
    return TensorMetadata(
        shape=shape,
        strides=strides,
        dtype=dtype,
        span=span,
        byte_offset=msg.byte_offset,
        dtype_code=msg.dtype_code,
        dtype_bits=msg.dtype_bits,
        dtype_lanes=msg.dtype_lanes,
    )


def _cuda_available() -> bool:
    try:
        _registry.for_device(torch.device('cuda'))
    except RuntimeError:
        return False
    return True


def allocate_tensor_msg(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]] = None,
) -> ExperimentalTensor:
    """Allocate a tensor message on CPU or CUDA and populate its metadata."""
    normalized_shape = list(shape)
    if any(dimension < 0 for dimension in normalized_shape):
        raise ValueError('Tensor shape dimensions must be nonnegative')
    if dtype not in _DTYPE_TO_DLPACK:
        raise TypeError(f'Unsupported torch dtype {dtype}')
    if device is None:
        selected_device = _registry.default_device()
    else:
        selected_device = torch.device(device)
    backend = _registry.for_device(selected_device)

    msg = ExperimentalTensor()
    dtype_code, dtype_bits, dtype_lanes = _DTYPE_TO_DLPACK[dtype]
    msg.dtype_code = dtype_code
    msg.dtype_bits = dtype_bits
    msg.dtype_lanes = dtype_lanes
    msg.shape = normalized_shape
    msg.strides = _contiguous_strides(normalized_shape)
    msg.byte_offset = 0
    byte_count = prod(normalized_shape) * (dtype_bits * dtype_lanes // 8)
    msg.data = backend.allocate(byte_count, selected_device)
    return msg


def from_output_tensor_msg(
    msg: ExperimentalTensor,
) -> Optional[torch.Tensor]:
    """Return a writable tensor view that owns its backend write handle."""
    if len(msg.data) == 0:
        return None
    metadata = _metadata(msg)
    return _registry.for_data(msg.data).from_output(msg.data, metadata)


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    clone: bool = True,
) -> Optional[torch.Tensor]:
    """Return an independent tensor or a zero-copy message view."""
    if len(msg.data) == 0:
        return None
    metadata = _metadata(msg)
    tensor = _registry.for_data(msg.data).from_input(msg.data, metadata)
    return tensor.clone() if clone else tensor


@overload
def to_tensor_msg(tensor: torch.Tensor) -> ExperimentalTensor:
    ...


@overload
def to_tensor_msg(
    msg: ExperimentalTensor,
    tensor: torch.Tensor,
) -> ExperimentalTensor:
    ...


def to_tensor_msg(*args: object) -> ExperimentalTensor:
    """Copy a tensor into a new message or into a supplied message."""
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
        msg = args[0]
        tensor = args[1]
        if tensor.numel() == 0:
            return msg
    else:
        raise TypeError(
            'Expected to_tensor_msg(tensor) or to_tensor_msg(msg, tensor)'
        )

    contiguous = tensor.contiguous()
    required_size = contiguous.numel() * contiguous.element_size()
    if required_size > len(msg.data):
        raise ValueError(
            f'Tensor requires {required_size} bytes, '
            f'but the buffer has {len(msg.data)}'
        )
    _set_metadata(msg, contiguous)
    output = from_output_tensor_msg(msg)
    if output is None:
        return msg
    output.copy_(contiguous)
    return msg


def set_stream(
    device: Optional[Union[str, torch.device]] = None,
):
    """Return a context manager selecting a backend stream when available."""
    if device is None:
        selected_device = _registry.default_device()
    else:
        selected_device = torch.device(device)
    return _registry.for_device(selected_device).stream_context()


__all__ = [
    'allocate_tensor_msg',
    'from_input_tensor_msg',
    'from_output_tensor_msg',
    'set_stream',
    'to_tensor_msg',
]
