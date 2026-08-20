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

"""PyTorch conversion adapter contract and dispatch."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol
from typing import Sequence

from rosidl_buffer import Buffer

import torch


@dataclass(frozen=True)
class TensorMetadata:
    """Validated tensor view metadata."""

    shape: Sequence[int]
    strides: Sequence[int]
    dtype: torch.dtype
    span: int
    byte_offset: int
    dtype_code: int
    dtype_bits: int
    dtype_lanes: int


class TorchConversionAdapter(Protocol):
    """Operations needed to convert storage to and from PyTorch."""

    device_type: str
    buffer_backend: str | None
    priority: int

    def is_available(self) -> bool:
        """Return whether the adapter can currently be used."""

    def matches(self, data: object) -> bool:
        """Return whether this adapter owns non-Buffer storage."""

    def allocate(self, byte_count: int, device: torch.device) -> object:
        """Allocate message storage."""

    def from_input(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor:
        """Create a readable tensor view."""

    def from_output(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor:
        """Create a writable tensor view."""

    def stream_context(self) -> AbstractContextManager:
        """Return a context selecting an adapter stream."""

    def unavailable_error(self) -> RuntimeError:
        """Describe why the adapter is unavailable."""


class TorchConversionRegistry:
    """Resolve conversion adapters by torch device or message storage."""

    def __init__(self) -> None:
        self._by_device: dict[str, TorchConversionAdapter] = {}
        self._by_buffer_backend: dict[str, TorchConversionAdapter] = {}
        self._fallbacks: list[TorchConversionAdapter] = []

    def register(self, adapter: TorchConversionAdapter) -> None:
        if adapter.device_type in self._by_device:
            raise ValueError(
                'Torch conversion adapter for device '
                f'{adapter.device_type!r} is already registered'
            )
        if (
            adapter.buffer_backend is not None
            and adapter.buffer_backend in self._by_buffer_backend
        ):
            raise ValueError(
                'Torch conversion adapter for buffer '
                f'{adapter.buffer_backend!r} is already registered'
            )

        self._by_device[adapter.device_type] = adapter
        if adapter.buffer_backend is None:
            self._fallbacks.append(adapter)
        else:
            self._by_buffer_backend[adapter.buffer_backend] = adapter

    def for_device(self, device: torch.device) -> TorchConversionAdapter:
        adapter = self._by_device.get(device.type)
        if adapter is None:
            raise ValueError(
                f'Unsupported tensor device {device.type!r}'
            )
        if not adapter.is_available():
            raise adapter.unavailable_error()
        return adapter

    def for_data(self, data: object) -> TorchConversionAdapter:
        if isinstance(data, Buffer):
            adapter = self._by_buffer_backend.get(data.backend_type)
            if adapter is None:
                raise ValueError(
                    f'Unsupported buffer backend {data.backend_type!r}'
                )
        else:
            adapter = next(
                (
                    candidate
                    for candidate in self._fallbacks
                    if candidate.matches(data)
                ),
                None,
            )
            if adapter is None:
                raise ValueError(
                    f'Unsupported tensor storage {type(data).__name__!r}'
                )

        if not adapter.is_available():
            raise adapter.unavailable_error()
        return adapter

    def default_device(self) -> torch.device:
        available = [
            adapter
            for adapter in self._by_device.values()
            if adapter.is_available()
        ]
        if not available:
            raise RuntimeError('No Torch conversion adapter is available')
        adapter = max(
            available,
            key=lambda item: (item.priority, item.device_type),
        )
        return torch.device(adapter.device_type)
