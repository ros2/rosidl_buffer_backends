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

"""CPU adapter for PyTorch conversions."""

from array import array
from contextlib import nullcontext

import torch

from torch_conversions._adapter import TensorMetadata


class CpuTorchConversionAdapter:
    """Provide PyTorch views over Python-owned CPU storage."""

    device_type = 'cpu'
    buffer_backend = None
    priority = 0

    def is_available(self) -> bool:
        return True

    def matches(self, data: object) -> bool:
        try:
            memoryview(data)
        except TypeError:
            return False
        return True

    def allocate(self, byte_count: int, device: torch.device) -> array:
        del device
        return array('B', bytes(byte_count))

    def from_input(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor:
        return self._view(data, metadata)

    def from_output(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor:
        return self._view(data, metadata)

    def stream_context(self):
        return nullcontext()

    def unavailable_error(self) -> RuntimeError:
        return RuntimeError('CPU PyTorch conversion support is unavailable')

    @staticmethod
    def _view(data: object, metadata: TensorMetadata) -> torch.Tensor:
        storage = torch.frombuffer(
            data,
            dtype=metadata.dtype,
            count=metadata.span,
            offset=metadata.byte_offset,
        )
        return torch.as_strided(storage, metadata.shape, metadata.strides)
