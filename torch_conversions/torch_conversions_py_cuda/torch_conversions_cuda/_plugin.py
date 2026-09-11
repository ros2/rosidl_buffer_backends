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
        return torch.cuda.is_available()

    def matches(self, data: object) -> bool:
        del data
        return False

    def allocate(self, byte_count: int, backend: str) -> object:
        del backend
        return CudaBuffer.allocate_buffer(byte_count)

    def from_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> torch.Tensor:
        return _tensor(
            CudaBuffer.from_input_buffer(data, stream), data, metadata)

    def from_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> torch.Tensor:
        return _tensor(
            CudaBuffer.from_output_buffer(data, stream), data, metadata)

    def copy_to(
        self,
        data: object,
        metadata: TensorMetadata,
        source: torch.Tensor,
        stream: Optional[int],
    ) -> None:
        self.from_output(data, metadata, stream).copy_(source)


class _Lease:
    """Keep the CUDA mapping and its backing buffer alive with the tensor."""

    def __init__(self, handle: object, data: object) -> None:
        self._handle = handle
        self._data = data

    def __del__(self) -> None:
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
