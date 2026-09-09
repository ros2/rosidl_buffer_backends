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
from contextlib import nullcontext
from importlib.util import find_spec

import torch

from torch_conversions._dlpack_bridge import make_dlpack_capsule
from torch_conversions._plugin import TensorMetadata
from torch_conversions._plugin import TorchConversionRegistry


_DL_CUDA = 2


def _require_torch_version() -> None:
    version = torch.__version__.split('+', 1)[0]
    if version != '2.9.1':
        raise RuntimeError(
            'torch_conversions requires PyTorch 2.9.1, '
            f'but imported {torch.__version__} from {torch.__file__}')


def _cuda_buffer_installed() -> bool:
    return find_spec('cuda_buffer') is not None


def _cuda_buffer():
    from cuda_buffer import CudaBuffer
    return CudaBuffer


class CpuTorchConversionPlugin:
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
        return torch.as_strided(
            storage, metadata.shape, metadata.strides
        )


class CudaTorchConversionPlugin:
    device_type = 'cuda'
    buffer_backend = 'cuda'
    priority = 100

    def is_available(self) -> bool:
        return _cuda_buffer_installed() and torch.cuda.is_available()

    def matches(self, data: object) -> bool:
        del data
        return False

    def allocate(self, byte_count: int, device: torch.device) -> object:
        self._require()
        device_context = (
            nullcontext()
            if device.index is None else torch.cuda.device(device)
        )
        with device_context:
            return _cuda_buffer().allocate_buffer(byte_count)

    def from_input(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor:
        self._require()
        return self._from_buffer(data, metadata, writable=False)

    def from_output(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor:
        self._require()
        return self._from_buffer(data, metadata, writable=True)

    @staticmethod
    def _from_buffer(
        data: object,
        metadata: TensorMetadata,
        writable: bool,
    ) -> torch.Tensor:
        cuda_buffer = _cuda_buffer()
        stream = torch.cuda.current_stream()
        factory = (
            cuda_buffer.from_output_buffer
            if writable else cuda_buffer.from_input_buffer
        )
        handle = factory(data, stream.cuda_stream)
        capsule = make_dlpack_capsule(
            handle.device_ptr,
            _DL_CUDA,
            torch.cuda.current_device(),
            metadata.dtype_code,
            metadata.dtype_bits,
            metadata.dtype_lanes,
            list(metadata.shape),
            list(metadata.strides),
            metadata.byte_offset,
            handle,
        )
        return torch.utils.dlpack.from_dlpack(capsule)

    def stream_context(self):
        self._require()
        return torch.cuda.stream(torch.cuda.Stream())

    def unavailable_error(self) -> RuntimeError:
        if not _cuda_buffer_installed():
            return RuntimeError(
                'CUDA conversion requires the cuda_buffer_py package'
            )
        return RuntimeError(
            'CUDA was requested but is not available to PyTorch'
        )

    def _require(self) -> None:
        if not self.is_available():
            raise self.unavailable_error()


def register(registry: TorchConversionRegistry) -> None:
    _require_torch_version()
    registry.register(CpuTorchConversionPlugin())
    registry.register(CudaTorchConversionPlugin())
