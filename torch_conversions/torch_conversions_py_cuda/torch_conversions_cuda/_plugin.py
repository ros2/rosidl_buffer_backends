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

from typing import Optional

from cuda_buffer import CudaBuffer

import torch
import torch.utils.dlpack

from torch_conversions._plugin import ConversionRegistry
from torch_conversions._plugin import TensorMetadata
from torch_conversions._torch_bridge import make_dlpack_capsule


class CudaConversionPlugin:
    """Direct Torch views over CUDA-backed tensor message storage."""

    backends = ('cuda',)
    device_types = ('cuda',)
    priority = 100

    def is_available(self) -> bool:
        return torch.version.cuda is not None and torch.cuda.is_available()

    def matches(self, data: object) -> bool:
        del data
        return False

    def allocate(self, byte_count: int, backend: str, device: torch.device) -> object:
        del backend
        with torch.cuda.device(device):
            data = CudaBuffer.allocate_buffer(byte_count)
            if byte_count:
                with CudaBuffer.from_input_buffer(data, 0) as handle:
                    _check_device(handle)
            return data

    def stream_context(self, device: torch.device):
        return torch.cuda.stream(torch.cuda.Stream(device=device))

    def from_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int],
        clone: bool,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        raw_stream = _stream_pointer(stream, device)
        handle = CudaBuffer.from_input_buffer(data, raw_stream or 1)
        _check_device(handle)
        selected = torch.cuda.ExternalStream(raw_stream, device=device)
        with torch.cuda.stream(selected):
            view = _tensor(handle, data, metadata)
            return view.clone() if clone else view

    def from_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        handle = CudaBuffer.from_output_buffer(data, _stream_pointer(stream, device) or 1)
        _check_device(handle)
        return _tensor(handle, data, metadata)

    def copy_to(
        self,
        data: object,
        metadata: TensorMetadata,
        source: torch.Tensor,
        stream: Optional[int],
    ) -> None:
        if getattr(data, 'backend_type', None) == 'cuda':
            destination = self.from_output(data, metadata, stream)
            device = destination.device
            if source.device.type != 'cpu' and source.device != device:
                raise ValueError('CUDA copies require matching source and destination devices')
        else:
            destination = torch.frombuffer(
                data, dtype=metadata.dtype, count=source.numel(),
                offset=metadata.byte_offset).reshape(metadata.shape)
            device = source.device
        selected = torch.cuda.ExternalStream(_stream_pointer(stream, device), device=device)
        with torch.cuda.stream(selected):
            destination.copy_(source.contiguous())
            selected.synchronize()


def _check_device(handle: object) -> None:
    current = torch.cuda.current_device()
    if handle.device_id != current:
        handle.close()
        raise RuntimeError(
            f'CUDA buffer is on device {handle.device_id}, but current device is {current}')


def _stream_pointer(stream: Optional[int], device: object) -> int:
    return torch.cuda.current_stream(device).cuda_stream if stream is None else stream


class _Lease:
    """Keep the CUDA mapping and its backing buffer alive with the tensor."""

    def __init__(self, handle: object, data: object) -> None:
        self._handle = handle
        self._data = data

    def __del__(self) -> None:
        with torch.cuda.device(self._handle.device_id):
            self._handle.close()


def _tensor(
    handle: object, data: object, metadata: TensorMetadata
) -> torch.Tensor:
    capsule = make_dlpack_capsule(
        handle.device_ptr + metadata.byte_offset,
        2,
        handle.device_id,
        metadata.dtype_code,
        metadata.dtype_bits,
        metadata.dtype_lanes,
        list(metadata.shape),
        list(metadata.strides),
        0,
        _Lease(handle, data),
    )
    return torch.utils.dlpack.from_dlpack(capsule)


def register(registry: ConversionRegistry) -> None:
    registry.register(CudaConversionPlugin())
