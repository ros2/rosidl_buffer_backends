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

"""Convert tensor_msgs/ExperimentalTensor messages to and from PyTorch tensors."""

from array import array
from contextlib import nullcontext
from math import prod
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Union

from rosidl_buffer import Buffer
from tensor_msgs.msg import ExperimentalTensor
import torch

try:
    from cuda_buffer import CudaBuffer
    from torch_conversions._torch_conversions_py import _from_input_dlpack
    from torch_conversions._torch_conversions_py import _from_output_dlpack
except ImportError as error:
    CudaBuffer = None
    _from_input_dlpack = None
    _from_output_dlpack = None
    _CUDA_IMPORT_ERROR = error
else:
    _CUDA_IMPORT_ERROR = None


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
) -> tuple[list[int], list[int], torch.dtype, int]:
    shape = list(msg.shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError('Tensor shape dimensions must be nonnegative')
    strides = list(msg.strides) or _contiguous_strides(shape)
    if len(strides) != len(shape):
        raise ValueError('Tensor strides must be empty or match the shape rank')
    if any(stride < 0 for stride in strides):
        raise ValueError('Negative tensor strides are unsupported')

    dtype = _dtype_from_msg(msg)
    element_size = torch.empty((), dtype=dtype).element_size()
    span = 0 if 0 in shape else 1
    if span:
        span += sum((dimension - 1) * stride for dimension, stride in zip(shape, strides))
    required_size = msg.byte_offset + span * element_size
    if required_size > len(msg.data):
        raise ValueError(
            f'Tensor view requires {required_size} bytes, but the buffer has {len(msg.data)}'
        )
    return shape, strides, dtype, span


def _is_cuda_buffer(data: object) -> bool:
    return isinstance(data, Buffer) and data.backend_type == 'cuda'


def _cuda_available() -> bool:
    return CudaBuffer is not None and torch.cuda.is_available()


def _require_cuda_support() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available to PyTorch')
    if CudaBuffer is None:
        raise RuntimeError(
            'CUDA buffer support was not built for torch_conversions'
        ) from _CUDA_IMPORT_ERROR


def _current_cuda_stream() -> int:
    return torch.cuda.current_stream().cuda_stream


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
    selected_device = torch.device(
        device if device is not None else ('cuda' if _cuda_available() else 'cpu')
    )
    if selected_device.type not in ('cpu', 'cuda'):
        raise ValueError(f'Unsupported tensor device {selected_device.type}')
    if selected_device.type == 'cuda':
        _require_cuda_support()

    msg = ExperimentalTensor()
    dtype_code, dtype_bits, dtype_lanes = _DTYPE_TO_DLPACK[dtype]
    msg.dtype_code = dtype_code
    msg.dtype_bits = dtype_bits
    msg.dtype_lanes = dtype_lanes
    msg.shape = normalized_shape
    msg.strides = _contiguous_strides(normalized_shape)
    msg.byte_offset = 0
    byte_count = prod(normalized_shape) * (dtype_bits * dtype_lanes // 8)
    if selected_device.type == 'cuda':
        msg.data = CudaBuffer.allocate_buffer(byte_count)
    else:
        msg.data = array('B', bytes(byte_count))
    return msg


def from_output_tensor_msg(msg: ExperimentalTensor) -> Optional[torch.Tensor]:
    """Return a writable tensor view whose lifetime owns the backend write handle."""
    if len(msg.data) == 0:
        return None
    shape, strides, dtype, span = _metadata(msg)
    if _is_cuda_buffer(msg.data):
        _require_cuda_support()
        capsule = _from_output_dlpack(
            msg.data,
            shape,
            strides,
            msg.dtype_code,
            msg.dtype_bits,
            msg.dtype_lanes,
            msg.byte_offset,
            _current_cuda_stream(),
        )
        return torch.utils.dlpack.from_dlpack(capsule)
    if isinstance(msg.data, Buffer):
        raise ValueError(f'Unsupported buffer backend {msg.data.backend_type!r}')
    storage = torch.frombuffer(
        msg.data,
        dtype=dtype,
        count=span,
        offset=msg.byte_offset,
    )
    return torch.as_strided(storage, shape, strides)


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    clone: bool = True,
) -> Optional[torch.Tensor]:
    """Return an independent tensor or a zero-copy read-only view of a message."""
    if len(msg.data) == 0:
        return None
    shape, strides, dtype, span = _metadata(msg)
    if _is_cuda_buffer(msg.data):
        _require_cuda_support()
        capsule = _from_input_dlpack(
            msg.data,
            shape,
            strides,
            msg.dtype_code,
            msg.dtype_bits,
            msg.dtype_lanes,
            msg.byte_offset,
            _current_cuda_stream(),
        )
        tensor = torch.utils.dlpack.from_dlpack(capsule)
    elif isinstance(msg.data, Buffer):
        raise ValueError(f'Unsupported buffer backend {msg.data.backend_type!r}')
    else:
        storage = torch.frombuffer(
            msg.data,
            dtype=dtype,
            count=span,
            offset=msg.byte_offset,
        )
        tensor = torch.as_strided(storage, shape, strides)
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
        raise TypeError('Expected to_tensor_msg(tensor) or to_tensor_msg(msg, tensor)')

    contiguous = tensor.contiguous()
    required_size = contiguous.numel() * contiguous.element_size()
    if required_size > len(msg.data):
        raise ValueError(
            f'Tensor requires {required_size} bytes, but the buffer has {len(msg.data)}'
        )
    _set_metadata(msg, contiguous)
    output = from_output_tensor_msg(msg)
    if output is None:
        return msg
    output.copy_(contiguous)
    return msg


def set_stream():
    """Return a context manager selecting a non-default CUDA stream when available."""
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.stream(torch.cuda.Stream())


__all__ = [
    'allocate_tensor_msg',
    'from_input_tensor_msg',
    'from_output_tensor_msg',
    'set_stream',
    'to_tensor_msg',
]
