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

"""Framework-free DLPack storage for ExperimentalTensor messages."""

from dlpack_conversions._core import allocate_tensor_msg
from dlpack_conversions._core import available_backends
from dlpack_conversions._core import backend_available
from dlpack_conversions._core import backend_for_device
from dlpack_conversions._core import contiguous_strides
from dlpack_conversions._core import default_backend
from dlpack_conversions._core import device_for_backend
from dlpack_conversions._core import from_input_tensor_msg
from dlpack_conversions._core import from_output_tensor_msg
from dlpack_conversions._core import metadata
from dlpack_conversions._plugin import CPU
from dlpack_conversions._plugin import CUDA
from dlpack_conversions._plugin import ROCM
from dlpack_conversions._plugin import StoragePlugin
from dlpack_conversions._plugin import StorageRegistry
from dlpack_conversions._plugin import TensorMetadata

__all__ = [
    'CPU',
    'CUDA',
    'ROCM',
    'StoragePlugin',
    'StorageRegistry',
    'TensorMetadata',
    'allocate_tensor_msg',
    'available_backends',
    'backend_available',
    'backend_for_device',
    'contiguous_strides',
    'default_backend',
    'device_for_backend',
    'from_input_tensor_msg',
    'from_output_tensor_msg',
    'metadata',
]
