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

"""CUDA adapter for PyTorch conversions."""

from contextlib import nullcontext

import torch

from torch_conversions._adapter import TensorMetadata

try:
    from cuda_buffer import CudaBuffer
    from torch_conversions._torch_conversions_py import _from_input_dlpack
    from torch_conversions._torch_conversions_py import _from_output_dlpack
except ImportError as error:
    CudaBuffer = None
    _from_input_dlpack = None
    _from_output_dlpack = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


class CudaTorchConversionAdapter:
    """Provide PyTorch DLPack views over CUDA-backed storage."""

    device_type = 'cuda'
    buffer_backend = 'cuda'
    priority = 100

    def is_available(self) -> bool:
        return CudaBuffer is not None and torch.cuda.is_available()

    def matches(self, data: object) -> bool:
        del data
        return False

    def allocate(self, byte_count: int, device: torch.device) -> object:
        self._require()
        if device.index is None:
            device_context = nullcontext()
        else:
            device_context = torch.cuda.device(device)
        with device_context:
            return CudaBuffer.allocate_buffer(byte_count)

    def from_input(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor:
        self._require()
        capsule = _from_input_dlpack(
            data,
            list(metadata.shape),
            list(metadata.strides),
            metadata.dtype_code,
            metadata.dtype_bits,
            metadata.dtype_lanes,
            metadata.byte_offset,
            self._current_stream(),
        )
        return torch.utils.dlpack.from_dlpack(capsule)

    def from_output(
        self, data: object, metadata: TensorMetadata
    ) -> torch.Tensor:
        self._require()
        capsule = _from_output_dlpack(
            data,
            list(metadata.shape),
            list(metadata.strides),
            metadata.dtype_code,
            metadata.dtype_bits,
            metadata.dtype_lanes,
            metadata.byte_offset,
            self._current_stream(),
        )
        return torch.utils.dlpack.from_dlpack(capsule)

    def stream_context(self):
        self._require()
        return torch.cuda.stream(torch.cuda.Stream())

    def unavailable_error(self) -> RuntimeError:
        if not torch.cuda.is_available():
            return RuntimeError(
                'CUDA was requested but is not available to PyTorch'
            )
        return RuntimeError(
            'CUDA buffer support was not built for torch_conversions'
        )

    def _require(self) -> None:
        if not self.is_available():
            raise self.unavailable_error() from _IMPORT_ERROR

    @staticmethod
    def _current_stream() -> int:
        return torch.cuda.current_stream().cuda_stream
