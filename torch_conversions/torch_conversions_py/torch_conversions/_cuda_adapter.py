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

"""Optional CUDA adapter for PyTorch conversions."""

from contextlib import nullcontext
from importlib.util import find_spec

import torch

from torch_conversions._adapter import TensorMetadata
from torch_conversions._dlpack_bridge import make_dlpack_capsule


_DL_CUDA = 2


def _cuda_buffer_installed() -> bool:
    try:
        return find_spec('cuda_buffer') is not None
    except (ImportError, ValueError):
        return False


def _cuda_buffer():
    try:
        from cuda_buffer import CudaBuffer
    except ImportError as error:
        raise RuntimeError(
            'CUDA conversion requires the optional cuda_buffer_py package'
        ) from error
    return CudaBuffer


class CudaTorchConversionAdapter:
    """Provide PyTorch DLPack views over optional CUDA-backed storage."""

    device_type = 'cuda'
    buffer_backend = 'cuda'
    priority = 100

    def is_available(self) -> bool:
        if not _cuda_buffer_installed():
            return False
        try:
            return torch.cuda.is_available()
        except RuntimeError:
            return False

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
            handle.device_id,
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
                'CUDA conversion requires the optional cuda_buffer_py package'
            )
        return RuntimeError(
            'CUDA was requested but is not available to PyTorch'
        )

    def _require(self) -> None:
        if not self.is_available():
            raise self.unavailable_error()
