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

from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol
from typing import Sequence

from ament_index_python.resources import get_resource
from ament_index_python.resources import get_resources
from rosidl_buffer import Buffer

import torch


_PLUGIN_RESOURCE_TYPE = 'torch_conversions__python_plugins'


@dataclass(frozen=True)
class TensorMetadata:
    shape: Sequence[int]
    strides: Sequence[int]
    dtype: torch.dtype
    span: int
    byte_offset: int
    dtype_code: int
    dtype_bits: int
    dtype_lanes: int


class TorchConversionPlugin(Protocol):
    device_type: str
    buffer_backend: str | None
    priority: int

    def is_available(self) -> bool: ...

    def matches(self, data: object) -> bool: ...

    def allocate(self, byte_count: int, device: torch.device) -> object: ...

    def from_input(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor: ...

    def from_output(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor: ...

    def stream_context(self) -> AbstractContextManager: ...

    def unavailable_error(self) -> RuntimeError: ...


class TorchConversionRegistry:

    def __init__(self) -> None:
        self._by_device: dict[str, TorchConversionPlugin] = {}
        self._by_buffer_backend: dict[str, TorchConversionPlugin] = {}
        self._fallbacks: list[TorchConversionPlugin] = []

    def register(self, plugin: TorchConversionPlugin) -> None:
        if plugin.device_type in self._by_device:
            raise ValueError(
                'Torch conversion plugin for device '
                f'{plugin.device_type!r} is already registered'
            )
        if (
            plugin.buffer_backend is not None
            and plugin.buffer_backend in self._by_buffer_backend
        ):
            raise ValueError(
                'Torch conversion plugin for storage '
                f'{plugin.buffer_backend!r} is already registered'
            )
        self._by_device[plugin.device_type] = plugin
        if plugin.buffer_backend is None:
            self._fallbacks.append(plugin)
        else:
            self._by_buffer_backend[plugin.buffer_backend] = plugin

    def for_device(self, device: torch.device) -> TorchConversionPlugin:
        plugin = self._by_device.get(device.type)
        if plugin is None:
            raise ValueError(f'Unsupported tensor device: {device.type!r}')
        if not plugin.is_available():
            raise plugin.unavailable_error()
        return plugin

    def for_data(self, data: object) -> TorchConversionPlugin:
        if isinstance(data, Buffer):
            plugin = self._by_buffer_backend.get(data.backend_type)
        else:
            plugin = next(
                (item for item in self._fallbacks if item.matches(data)),
                None,
            )
        if plugin is None:
            backend = getattr(data, 'backend_type', type(data).__name__)
            raise ValueError(f'Unsupported tensor storage: {backend!r}')
        if not plugin.is_available():
            raise plugin.unavailable_error()
        return plugin

    def default_device(self) -> torch.device:
        available = [
            plugin for plugin in self._by_device.values()
            if plugin.is_available()
        ]
        if not available:
            raise RuntimeError(
                'No Torch Python conversion runtime is installed'
            )
        return torch.device(max(
            available,
            key=lambda item: (item.priority, item.device_type),
        ).device_type)


def load_external_plugins(registry: TorchConversionRegistry) -> None:
    resources = get_resources(_PLUGIN_RESOURCE_TYPE)
    if len(resources) != 1:
        raise RuntimeError(
            'torch_conversions requires exactly one Python runtime plugin; '
            f'found {len(resources)}: {sorted(resources)}'
        )
    package_name = next(iter(resources))
    content, _ = get_resource(_PLUGIN_RESOURCE_TYPE, package_name)
    module_name, separator, function_name = content.strip().partition(':')
    if not separator or not module_name or not function_name:
        raise RuntimeError(
            f'Invalid Torch conversion plugin resource from {package_name}'
        )
    getattr(import_module(module_name), function_name)(registry)
