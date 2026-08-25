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

"""Register CUDA support for torch_conversions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_conversions._adapter import TorchConversionRegistry


def register(registry: 'TorchConversionRegistry') -> None:
    """Register the CUDA conversion adapter."""
    from torch_conversions_cuda_plugin._cuda_adapter import (
        CudaTorchConversionAdapter
    )

    registry.register(CudaTorchConversionAdapter())


__all__ = ['register']
