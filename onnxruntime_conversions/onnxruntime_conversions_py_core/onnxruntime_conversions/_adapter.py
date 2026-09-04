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

"""ONNX Runtime conversion adapter contract and dispatch."""

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


_ADAPTER_RESOURCE_TYPE = 'onnxruntime_conversions__adapters'


@dataclass(frozen=True)
class TensorMetadata:
    """Validated tensor metadata shared with conversion adapters."""

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


class OrtConversionAdapter(Protocol):
    """Operations provided by an ONNX Runtime storage adapter."""

    device_type: str
    buffer_backend: str
    priority: int

    def is_available(self) -> bool:
        """Return whether the adapter runtime can be used."""

    def unavailable_error(self) -> RuntimeError:
        """Describe why the adapter runtime cannot be used."""

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
    """Resolve adapters by requested device or message storage."""

    def __init__(self) -> None:
        self._by_device: dict[str, OrtConversionAdapter] = {}
        self._by_buffer_backend: dict[str, OrtConversionAdapter] = {}

    def register(self, adapter: OrtConversionAdapter) -> None:
        if adapter.device_type in self._by_device:
            raise ValueError(
                f'Adapter for device {adapter.device_type!r} is registered')
        if adapter.buffer_backend in self._by_buffer_backend:
            raise ValueError(
                'Adapter for buffer backend '
                f'{adapter.buffer_backend!r} is registered')
        self._by_device[adapter.device_type] = adapter
        self._by_buffer_backend[adapter.buffer_backend] = adapter

    def for_device(self, device_type: str) -> OrtConversionAdapter:
        adapter = self._by_device.get(device_type)
        if adapter is None:
            raise ValueError(f'Unsupported tensor device type: {device_type}')
        return self._require_available(adapter)

    def for_data(self, data: object) -> OrtConversionAdapter:
        backend = str(getattr(data, 'backend_type', 'cpu')).lower()
        adapter = self._by_buffer_backend.get(backend)
        if adapter is None:
            raise ValueError(f'Unsupported tensor buffer backend: {backend}')
        return self._require_available(adapter)

    def default(self) -> OrtConversionAdapter:
        priority = max(
            adapter.priority for adapter in self._by_device.values())
        candidates = [
            adapter for adapter in self._by_device.values()
            if adapter.priority == priority
        ]
        if len(candidates) != 1:
            devices = ', '.join(
                sorted(adapter.device_type for adapter in candidates))
            raise RuntimeError(
                f'Ambiguous default ONNX Runtime adapters: {devices}')
        return self._require_available(candidates[0])

    @staticmethod
    def _require_available(
        adapter: OrtConversionAdapter,
    ) -> OrtConversionAdapter:
        if not adapter.is_available():
            raise adapter.unavailable_error()
        return adapter


def load_external_adapters(registry: OrtConversionRegistry) -> None:
    """Load adapters advertised through the ament resource index."""
    resources = get_resources(_ADAPTER_RESOURCE_TYPE)
    for package_name in sorted(resources):
        content, _ = get_resource(_ADAPTER_RESOURCE_TYPE, package_name)
        module_name, separator, function_name = content.strip().partition(':')
        if not separator or not module_name or not function_name:
            raise RuntimeError(
                f'Invalid ONNX Runtime adapter resource from {package_name}')
        register = getattr(import_module(module_name), function_name)
        register(registry)
