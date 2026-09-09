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
from typing import Tuple

from dlpack_conversions._dlpack_bridge import capsule_tensor
from dlpack_conversions._plugin import load_external_plugins
from dlpack_conversions._plugin import StorageRegistry
from dlpack_conversions._plugin import TensorMetadata

from tensor_msgs.msg import ExperimentalTensor


DType = Tuple[int, int, int]

_registry = StorageRegistry()
load_external_plugins(_registry)


def contiguous_strides(shape: Sequence[int]) -> list[int]:
    strides = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        stride *= shape[index]
    return strides


def _item_size(dtype: DType) -> int:
    _, bits, lanes = dtype
    return (bits * lanes + 7) // 8


def metadata(
    msg: ExperimentalTensor,
    dtype: Optional[DType] = None,
) -> TensorMetadata:
    """Describe the view a message defines over its storage.

    Pass ``dtype`` to describe the storage as an equally sized dtype, which
    frameworks need when they cannot consume the message dtype directly.
    """
    shape = list(msg.shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError('Tensor shape dimensions must be nonnegative')
    strides = list(msg.strides) or contiguous_strides(shape)
    if len(strides) != len(shape) or any(stride < 0 for stride in strides):
        raise ValueError('Tensor strides must match rank and be nonnegative')
    stored = (msg.dtype_code, msg.dtype_bits, msg.dtype_lanes)
    if dtype is not None and _item_size(dtype) != _item_size(stored):
        raise ValueError(
            f'Cannot describe a {_item_size(stored)} byte dtype as '
            f'{dtype}, which is {_item_size(dtype)} bytes wide'
        )
    dtype = stored if dtype is None else dtype
    item_size = _item_size(dtype)
    span = 0 if 0 in shape else 1 + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(shape, strides)
    )
    required_size = msg.byte_offset + span * item_size
    if required_size > len(msg.data):
        raise ValueError(
            f'Tensor view needs {required_size} bytes; '
            f'buffer has {len(msg.data)}'
        )
    return TensorMetadata(
        shape, strides, dtype[0], dtype[1], dtype[2],
        span, msg.byte_offset, item_size,
    )


def available_backends() -> list[str]:
    return _registry.backends()


def default_backend() -> str:
    return _registry.default_backend()


def backend_for_device(dl_device_type: int) -> Optional[str]:
    return _registry.backend_for_device(dl_device_type)


def device_for_backend(backend: str) -> int:
    return _registry.device_for_backend(backend)


def backend_available(backend: str) -> bool:
    return backend in _registry.backends()


def allocate_tensor_msg(
    shape: Sequence[int],
    dtype: DType,
    backend: Optional[str] = None,
) -> ExperimentalTensor:
    normalized_shape = list(shape)
    if any(dimension < 0 for dimension in normalized_shape):
        raise ValueError('Tensor shape dimensions must be nonnegative')
    selected = _registry.default_backend() if backend is None else backend
    plugin = _registry.for_backend(selected)

    msg = ExperimentalTensor()
    msg.dtype_code, msg.dtype_bits, msg.dtype_lanes = dtype
    msg.shape = normalized_shape
    msg.strides = contiguous_strides(normalized_shape)
    msg.byte_offset = 0
    msg.data = plugin.allocate(
        prod(normalized_shape) * _item_size(dtype), selected
    )
    return msg


def from_input_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
    dtype: Optional[DType] = None,
) -> Optional[object]:
    if len(msg.data) == 0:
        return None
    plugin = _registry.for_data(msg.data)
    return plugin.acquire_input(msg.data, metadata(msg, dtype), stream)


def from_output_tensor_msg(
    msg: ExperimentalTensor,
    stream: Optional[int] = None,
    dtype: Optional[DType] = None,
) -> Optional[object]:
    if len(msg.data) == 0:
        return None
    plugin = _registry.for_data(msg.data)
    return plugin.acquire_output(msg.data, metadata(msg, dtype), stream)


def to_tensor_msg(
    destination_or_source: object,
    source: Optional[object] = None,
    stream: Optional[int] = None,
    backend: Optional[str] = None,
) -> ExperimentalTensor:
    """Copy a DLPack capsule into message storage and stamp its metadata.

    Called with one argument, allocates a message on the backend that owns the
    capsule. Called with two, copies into the message given first. The capsule
    is described rather than consumed, so its owner keeps it.
    """
    if source is None:
        source, destination = destination_or_source, None
    else:
        destination = destination_or_source
        if not isinstance(destination, ExperimentalTensor):
            raise TypeError('destination must be an ExperimentalTensor')

    described = capsule_tensor(source)
    shape = list(described['shape'])
    strides = list(described['strides'])
    if strides and strides != contiguous_strides(shape):
        raise ValueError('Cannot copy from a non-contiguous DLPack tensor')
    dtype = (
        described['dtype_code'],
        described['dtype_bits'],
        described['dtype_lanes'],
    )
    byte_count = prod(shape) * _item_size(dtype)

    source_backend = _registry.backend_for_device(described['device_type'])
    if source_backend is None:
        raise ValueError(
            'No storage plugin handles DLPack device type '
            f'{described["device_type"]}'
        )

    if destination is None:
        destination = allocate_tensor_msg(
            shape, dtype, backend or source_backend)
    elif byte_count > len(destination.data):
        raise ValueError(
            f'Source tensor needs {byte_count} bytes; destination buffer has '
            f'{len(destination.data)}'
        )

    # Only the accelerator's own plugin can read its device memory, so it also
    # copies into host-backed messages.
    copier = _registry.backend_of(destination.data) \
        if source_backend == 'cpu' else source_backend
    _registry.for_backend(copier).copy_to(
        destination.data,
        described['data'] + described['byte_offset'],
        byte_count,
        source_backend,
        stream,
    )

    destination.dtype_code, destination.dtype_bits, destination.dtype_lanes = \
        dtype
    destination.shape = shape
    destination.strides = contiguous_strides(shape)
    destination.byte_offset = 0
    return destination
