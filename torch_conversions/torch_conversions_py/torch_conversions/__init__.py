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

"""Convert ExperimentalTensor messages to and from PyTorch tensors."""

from torch_conversions import _core
from torch_conversions._core import allocate_tensor_msg
from torch_conversions._core import available_backends
from torch_conversions._core import backend_available
from torch_conversions._core import backend_for_device
from torch_conversions._core import default_backend
from torch_conversions._core import from_input_tensor_msg
from torch_conversions._core import from_output_tensor_msg
from torch_conversions._core import set_stream
from torch_conversions._core import to_tensor_msg


_plugin_available = _core._plugin_available

__all__ = [
    'allocate_tensor_msg',
    'available_backends',
    'backend_available',
    'backend_for_device',
    'default_backend',
    'from_input_tensor_msg',
    'from_output_tensor_msg',
    'set_stream',
    'to_tensor_msg',
]
