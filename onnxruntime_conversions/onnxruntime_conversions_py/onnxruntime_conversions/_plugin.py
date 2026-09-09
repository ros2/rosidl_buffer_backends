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

"""ONNX Runtime conversion plugin contract and dispatch."""

from dataclasses import dataclass
from importlib import import_module
from types import TracebackType
from typing import Optional
from typing import Protocol
from typing import Type

from ament_index_python.resources import get_resource
from ament_index_python.resources import get_resources
import onnxruntime as ort
from tensor_msgs.msg import ExperimentalTensor


_PLUGIN_RESOURCE_TYPE = 'onnxruntime_conversions__python_plugins'


@dataclass(frozen=True)
class TensorMetadata:
    """Validated tensor metadata shared with conversion plugins."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    element_type: int
    dtype_code: int
    dtype_bits: int
    dtype_lanes: int
    element_count: int
    byte_count: int
    byte_offset: int
    numpy_dtype: object


class OrtTensorView:
    """Own an OrtValue and every object backing its external storage."""

    def __init__(
        self,
        value: ort.OrtValue,
        message: ExperimentalTensor,
        storage_view: object,
        backend_handle: object = None,
    ) -> None:
        self._value: Optional[ort.OrtValue] = value
        self._message: Optional[ExperimentalTensor] = message
        self._storage_view = storage_view
        self._backend_handle = backend_handle

    @property
    def value(self) -> ort.OrtValue:
        if self._value is None:
            raise RuntimeError('OrtTensorView is closed')
        return self._value

    @property
    def closed(self) -> bool:
        return self._value is None

    def close(self) -> None:
        if self.closed:
            return
        self._value = None
        self._storage_view = None
        handle = self._backend_handle
        self._backend_handle = None
        if handle is not None:
            handle.close()
        self._message = None

    def __enter__(self) -> ort.OrtValue:
        return self.value

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self.close()
        return False


class OrtConversionPlugin(Protocol):
    """Operations provided by an ONNX Runtime conversion plugin."""

    device_type: str
    buffer_backend: str
    priority: int

    def is_available(self) -> bool:
        """Return whether the plugin runtime can be used."""

    def unavailable_error(self) -> RuntimeError:
        """Describe why the plugin runtime cannot be used."""

    def allocate(
        self,
        metadata: TensorMetadata,
        device_id: int,
        stream: Optional[int],
    ) -> object:
        """Allocate message storage."""

    def view(
        self,
        message: ExperimentalTensor,
        metadata: TensorMetadata,
        stream: Optional[int],
        output: bool,
    ) -> OrtTensorView:
        """Create an ONNX Runtime tensor view."""


class OrtConversionRegistry:
    """Resolve plugins by requested device or message storage."""

    def __init__(self) -> None:
        self._by_device: dict[str, OrtConversionPlugin] = {}
        self._by_buffer_backend: dict[str, OrtConversionPlugin] = {}

    def register(self, plugin: OrtConversionPlugin) -> None:
        if plugin.device_type in self._by_device:
            raise ValueError(
                f'Plugin for device {plugin.device_type!r} is registered')
        if plugin.buffer_backend in self._by_buffer_backend:
            raise ValueError(
                'Plugin for buffer backend '
                f'{plugin.buffer_backend!r} is registered')
        self._by_device[plugin.device_type] = plugin
        self._by_buffer_backend[plugin.buffer_backend] = plugin

    def for_device(self, device_type: str) -> OrtConversionPlugin:
        plugin = self._by_device.get(device_type)
        if plugin is None:
            raise ValueError(f'Unsupported tensor device type: {device_type}')
        return self._require_available(plugin)

    def for_data(self, data: object) -> OrtConversionPlugin:
        backend = str(getattr(data, 'backend_type', 'cpu')).lower()
        plugin = self._by_buffer_backend.get(backend)
        if plugin is None:
            raise ValueError(f'Unsupported tensor buffer backend: {backend}')
        return self._require_available(plugin)

    def default(self) -> OrtConversionPlugin:
        priority = max(
            plugin.priority for plugin in self._by_device.values())
        candidates = [
            plugin for plugin in self._by_device.values()
            if plugin.priority == priority
        ]
        if len(candidates) != 1:
            devices = ', '.join(
                sorted(plugin.device_type for plugin in candidates))
            raise RuntimeError(
                f'Ambiguous default ONNX Runtime plugins: {devices}')
        return self._require_available(candidates[0])

    @staticmethod
    def _require_available(
        plugin: OrtConversionPlugin,
    ) -> OrtConversionPlugin:
        if not plugin.is_available():
            raise plugin.unavailable_error()
        return plugin


def load_external_plugins(registry: OrtConversionRegistry) -> None:
    """Load the selected plugin advertised through the ament resource index."""
    resources = get_resources(_PLUGIN_RESOURCE_TYPE)
    if len(resources) != 1:
        raise RuntimeError(
            'onnxruntime_conversions requires exactly one Python runtime '
            f'plugin; found {len(resources)}: {sorted(resources)}')
    package_name = next(iter(resources))
    content, _ = get_resource(_PLUGIN_RESOURCE_TYPE, package_name)
    module_name, separator, function_name = content.strip().partition(':')
    if not separator or not module_name or not function_name:
        raise RuntimeError(
            f'Invalid ONNX Runtime plugin resource from {package_name}')
    register = getattr(import_module(module_name), function_name)
    register(registry)
