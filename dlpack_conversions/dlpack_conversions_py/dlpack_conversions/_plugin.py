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

from dataclasses import dataclass
from importlib import import_module
import os
from typing import Optional
from typing import Protocol
from typing import Sequence

from ament_index_python.resources import get_resource
from ament_index_python.resources import get_resources
from rosidl_buffer import Buffer


PLUGIN_RESOURCE_TYPE = 'dlpack_conversions__python_plugins'
BACKEND_ENVIRONMENT_VARIABLE = 'ROSIDL_TENSOR_BACKEND'

# DLPack device type codes. Plugins import these from here rather than from
# the package root, which is still initializing while they are loaded.
CPU = 1
CUDA = 2
ROCM = 10


@dataclass(frozen=True)
class TensorMetadata:
    shape: Sequence[int]
    strides: Sequence[int]
    dtype_code: int
    dtype_bits: int
    dtype_lanes: int
    span: int
    byte_offset: int
    item_size: int


class StoragePlugin(Protocol):
    """Allocates and exposes buffer storage for one accelerator."""

    backends: Sequence[str]
    dl_device_types: Sequence[int]
    priority: int

    def is_available(self) -> bool: ...

    def matches(self, data: object) -> bool: ...

    def allocate(self, byte_count: int, backend: str) -> object: ...

    def acquire_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> object: ...

    def acquire_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> object: ...

    def unavailable_error(self) -> RuntimeError: ...


class StorageRegistry:

    def __init__(self) -> None:
        self._by_backend: dict[str, StoragePlugin] = {}
        self._by_device: dict[int, str] = {}
        self._fallbacks: list[StoragePlugin] = []

    def register(self, plugin: StoragePlugin) -> None:
        if not plugin.is_available():
            return
        for backend in plugin.backends:
            if backend in self._by_backend:
                raise ValueError(
                    f'Storage backend {backend!r} is already registered'
                )
            self._by_backend[backend] = plugin
        for device_type in plugin.dl_device_types:
            self._by_device.setdefault(device_type, plugin.backends[0])
        if getattr(plugin, 'is_fallback', False):
            self._fallbacks.append(plugin)

    def backends(self) -> list[str]:
        return sorted(self._by_backend)

    def backend_for_device(self, dl_device_type: int) -> Optional[str]:
        return self._by_device.get(dl_device_type)

    def device_for_backend(self, backend: str) -> int:
        return self.for_backend(backend).dl_device_types[0]

    def default_backend(self) -> str:
        requested = os.environ.get(BACKEND_ENVIRONMENT_VARIABLE)
        if requested:
            if requested not in self._by_backend:
                raise RuntimeError(
                    f'{BACKEND_ENVIRONMENT_VARIABLE} requests {requested!r}, '
                    f'which is not available; installed backends are '
                    f'{self.backends()}'
                )
            return requested
        if not self._by_backend:
            raise RuntimeError('No DLPack storage plugin is installed')
        return max(
            sorted(self._by_backend),
            key=lambda name: self._by_backend[name].priority,
        )

    def for_backend(self, backend: str) -> StoragePlugin:
        plugin = self._by_backend.get(backend)
        if plugin is None:
            raise ValueError(
                f'No storage plugin provides backend {backend!r}; '
                f'installed backends are {self.backends()}'
            )
        return plugin

    def for_data(self, data: object) -> StoragePlugin:
        if isinstance(data, Buffer):
            plugin = self._by_backend.get(data.backend_type)
        else:
            plugin = next(
                (item for item in self._fallbacks if item.matches(data)), None
            )
        if plugin is None:
            backend = getattr(data, 'backend_type', type(data).__name__)
            raise ValueError(f'Unsupported tensor storage: {backend!r}')
        return plugin


def load_external_plugins(registry: StorageRegistry) -> None:
    for package_name in sorted(get_resources(PLUGIN_RESOURCE_TYPE)):
        content, _ = get_resource(PLUGIN_RESOURCE_TYPE, package_name)
        module_name, separator, function_name = content.strip().partition(':')
        if not separator or not module_name or not function_name:
            raise RuntimeError(
                f'Invalid storage plugin resource from {package_name}'
            )
        getattr(import_module(module_name), function_name)(registry)
