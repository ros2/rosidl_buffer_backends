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


PLUGIN_RESOURCE_TYPE = 'torch_conversions__python_plugins'
BACKEND_ENVIRONMENT_VARIABLE = 'ROSIDL_TENSOR_BACKEND'


@dataclass(frozen=True)
class TensorMetadata:
    shape: Sequence[int]
    strides: Sequence[int]
    dtype: object
    dtype_code: int
    dtype_bits: int
    dtype_lanes: int
    span: int
    byte_offset: int
    item_size: int


class ConversionPlugin(Protocol):
    """Torch conversion implementation for one device type."""

    backends: Sequence[str]
    device_types: Sequence[str]
    priority: int

    def is_available(self) -> bool: ...

    def matches(self, data: object) -> bool: ...

    def allocate(self, byte_count: int, backend: str, device: object) -> object: ...

    def stream_context(self, device: object) -> object: ...

    def from_input(
        self, data: object, metadata: TensorMetadata, stream: Optional[int],
        clone: bool,
    ) -> object: ...

    def from_output(
        self, data: object, metadata: TensorMetadata, stream: Optional[int]
    ) -> object: ...

    def copy_to(
        self,
        data: object,
        metadata: TensorMetadata,
        source: object,
        stream: Optional[int],
    ) -> None: ...


class ConversionRegistry:

    def __init__(self) -> None:
        self._by_backend: dict[str, ConversionPlugin] = {}
        self._by_device: dict[str, str] = {}
        self._fallbacks: list[ConversionPlugin] = []

    def register(self, plugin: ConversionPlugin) -> None:
        if not plugin.is_available():
            return
        for backend in plugin.backends:
            if backend in self._by_backend:
                raise ValueError(
                    f'Torch backend {backend!r} is already registered')
            self._by_backend[backend] = plugin
        for device_type in plugin.device_types:
            self._by_device.setdefault(device_type, plugin.backends[0])
        if getattr(plugin, 'is_fallback', False):
            self._fallbacks.append(plugin)

    def backends(self) -> list[str]:
        return sorted(self._by_backend)

    def backend_for_device(self, device_type: str) -> Optional[str]:
        return self._by_device.get(device_type)

    def default_backend(self) -> str:
        requested = os.environ.get(BACKEND_ENVIRONMENT_VARIABLE)
        if requested:
            return self.for_backend(requested).backends[0]
        if not self._by_backend:
            raise RuntimeError('No Torch conversion plugin is installed')
        return max(
            sorted(self._by_backend),
            key=lambda name: self._by_backend[name].priority,
        )

    def for_backend(self, backend: str) -> ConversionPlugin:
        plugin = self._by_backend.get(backend)
        if plugin is None:
            raise RuntimeError(
                f'No Torch conversion plugin provides backend {backend!r}; '
                f'installed backends are {self.backends()}')
        return plugin

    def for_device(self, device_type: str) -> ConversionPlugin:
        backend = self.backend_for_device(device_type)
        if backend is None:
            raise RuntimeError(
                f'No Torch conversion plugin serves device {device_type!r}; '
                f'installed backends are {self.backends()}')
        return self.for_backend(backend)

    def for_data(self, data: object) -> ConversionPlugin:
        if isinstance(data, Buffer):
            plugin = self._by_backend.get(data.backend_type)
        else:
            plugin = next(
                (item for item in self._fallbacks if item.matches(data)), None)
        if plugin is None:
            backend = getattr(data, 'backend_type', type(data).__name__)
            raise RuntimeError(f'Unsupported tensor storage: {backend!r}')
        return plugin


def load_external_plugins(registry: ConversionRegistry) -> None:
    for package_name in sorted(get_resources(PLUGIN_RESOURCE_TYPE)):
        content, _ = get_resource(PLUGIN_RESOURCE_TYPE, package_name)
        module_name, separator, function_name = content.strip().partition(':')
        if not separator or not module_name or not function_name:
            raise RuntimeError(
                f'Invalid Torch conversion plugin resource from {package_name}')
        try:
            plugin_module = import_module(module_name)
        except (ImportError, OSError):
            continue
        getattr(plugin_module, function_name)(registry)
