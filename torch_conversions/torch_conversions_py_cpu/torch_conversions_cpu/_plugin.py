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

from array import array
from typing import Optional

import torch

from torch_conversions._plugin import ConversionRegistry
from torch_conversions._plugin import TensorMetadata


class CpuConversionPlugin:
    """Direct Torch views over host-backed tensor message storage."""

    backends = ('cpu',)
    device_types = ('cpu',)
    priority = 0
    is_fallback = True

    def is_available(self) -> bool:
        return True

    def matches(self, data: object) -> bool:
        try:
            memoryview(data)
        except TypeError:
            return False
        return True

    def allocate(self, byte_count: int, backend: str) -> array:
        del backend
        return array('B', bytes(byte_count))

    @staticmethod
    def _view(data: object, metadata: TensorMetadata) -> torch.Tensor:
        flat = torch.frombuffer(
            data,
            dtype=metadata.dtype,
            count=metadata.span,
            offset=metadata.byte_offset,
        )
        return torch.as_strided(
            flat, tuple(metadata.shape), tuple(metadata.strides))

    def from_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> torch.Tensor:
        del stream
        return self._view(data, metadata)

    def from_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> torch.Tensor:
        del stream
        return self._view(data, metadata)

    def copy_to(
        self,
        data: object,
        metadata: TensorMetadata,
        source: torch.Tensor,
        stream: Optional[int],
    ) -> None:
        del stream
        self._view(data, metadata).copy_(source)


def register(registry: ConversionRegistry) -> None:
    registry.register(CpuConversionPlugin())
