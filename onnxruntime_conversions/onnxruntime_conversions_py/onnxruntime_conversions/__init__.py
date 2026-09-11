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

"""ONNX Runtime views over tensor message storage."""

from onnxruntime_conversions._core import allocate_tensor_msg
from onnxruntime_conversions._core import available_backends
from onnxruntime_conversions._core import default_backend
from onnxruntime_conversions._core import from_input_tensor_msg
from onnxruntime_conversions._core import from_output_tensor_msg
from onnxruntime_conversions._core import OrtTensorView
from onnxruntime_conversions._core import session_providers
from onnxruntime_conversions._core import to_tensor_msg

__all__ = [
    'OrtTensorView',
    'allocate_tensor_msg',
    'available_backends',
    'default_backend',
    'from_input_tensor_msg',
    'from_output_tensor_msg',
    'session_providers',
    'to_tensor_msg',
]
