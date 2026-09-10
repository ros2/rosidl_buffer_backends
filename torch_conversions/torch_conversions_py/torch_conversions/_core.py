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

from contextlib import nullcontext
from typing import Optional
from typing import Sequence
from typing import Union

import dlpack_conversions

from tensor_msgs.msg import ExperimentalTensor

import torch
import torch.utils.dlpack


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

Device = Union[str, torch.device]


def _dl_device_type(device: torch.device) -> int:
    if device.type == 'cpu':
        return dlpack_conversions.CPU
    if device.type == 'cuda':
        return (
            dlpack_conversions.ROCM if torch.version.hip
            else dlpack_conversions.CUDA
        )
    raise ValueError(f'Unsupported tensor device: {device.type!r}')


def _backend_for(device: Device) -> str:
    normalized = torch.device(device)
    backend = dlpack_conversions.backend_for_device(
        _dl_device_type(normalized)
    )
    if backend is None:
        raise RuntimeError(
            f'No storage plugin serves {normalized.type!r} tensors; '
            f'installed backends are {dlpack_conversions.available_backends()}'
        )
    return backend


def _accelerator_available() -> bool:
    return torch.cuda.is_available() and _has_accelerator_backend()


def _default_backend() -> Optional[str]:
    """
    Choose a default backend this torch build can actually use.

    Storage plugins and the torch build are installed separately, so the core
    default can name a device this build has no kernels for. Handing that
    storage back would only fail later inside torch.
    """
    backend = dlpack_conversions.default_backend()
    if backend == 'cpu' or _accelerator_available():
        return backend
    available = dlpack_conversions.available_backends()
    return 'cpu' if 'cpu' in available else backend


def _has_accelerator_backend() -> bool:
    return any(
        backend != 'cpu'
        for backend in dlpack_conversions.available_backends()
    )


def _resolve_stream(stream: Optional[int]) -> Optional[int]:
    """
    Take the caller's stream, or fall back to the one torch is running on.

    The C++ adapter cannot do this: reading the current stream there means
    compiling against a CUDA LibTorch, which would decide a consumer's
    accelerator at build time. Here it is a plain runtime attribute lookup, so
    an explicit stream is only needed to override the current one.
    """
    if stream is not None:
        return stream
    if not _accelerator_available():
        return None
    return torch.cuda.current_stream().cuda_stream


def _plugin_available(device: Device) -> bool:
    try:
        _backend_for(device)
    except (RuntimeError, ValueError):
        return False
    return True


def allocate_tensor_msg(
    shape: Sequence[int],
    dtype: torch.dtype,
    device: Optional[Device] = None,
) -> ExperimentalTensor:
    if dtype not in _DTYPE_TO_DLPACK:
        raise TypeError(f'Unsupported torch dtype {dtype}')
    backend = _default_backend() if device is None else _backend_for(device)
    return dlpack_conversions.allocate_tensor_msg(
        shape, _DTYPE_TO_DLPACK[dtype], backend
    )


def from_output_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
) -> Optional[torch.Tensor]:
    capsule = dlpack_conversions.from_output_tensor_msg(
        msg, _resolve_stream(stream)
    )
    if capsule is None:
        return None
    return torch.utils.dlpack.from_dlpack(capsule)


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    clone: bool = True,
    stream: Optional[int] = None,
) -> Optional[torch.Tensor]:
    capsule = dlpack_conversions.from_input_tensor_msg(
        msg, _resolve_stream(stream)
    )
    if capsule is None:
        return None
    tensor = torch.utils.dlpack.from_dlpack(capsule)
    return tensor.clone() if clone else tensor


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
            'Expected to_tensor_msg(tensor) or to_tensor_msg(msg, tensor)'
        )
    contiguous = tensor.contiguous()
    if contiguous.dtype not in _DTYPE_TO_DLPACK:
        raise TypeError(f'Unsupported torch dtype {contiguous.dtype}')
    required_size = contiguous.numel() * contiguous.element_size()
    if required_size > len(msg.data):
        raise ValueError('Tensor exceeds allocated message storage')
    dtype = _DTYPE_TO_DLPACK[contiguous.dtype]
    msg.dtype_code, msg.dtype_bits, msg.dtype_lanes = dtype
    msg.shape = list(contiguous.shape)
    msg.strides = dlpack_conversions.contiguous_strides(msg.shape)
    msg.byte_offset = 0
    output = from_output_tensor_msg(msg, stream)
    if output is not None:
        output.copy_(contiguous)
    return msg


def set_stream(device: Optional[Device] = None):
    if device is not None and torch.device(device).type == 'cpu':
        return nullcontext()
    if not _accelerator_available():
        return nullcontext()
    return torch.cuda.stream(torch.cuda.Stream())
