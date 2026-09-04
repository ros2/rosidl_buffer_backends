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

"""CPU adapter for Python ONNX Runtime conversions."""

from array import array
from typing import Optional

import numpy as np
import onnxruntime as ort

from onnxruntime_conversions._adapter import OrtTensorView
from onnxruntime_conversions._adapter import TensorMetadata
from tensor_msgs.msg import ExperimentalTensor


class CpuOrtConversionAdapter:
    """Create ONNX Runtime views over CPU message storage."""

    device_type = 'cpu'
    buffer_backend = 'cpu'
    priority = 0

    def is_available(self) -> bool:
        return 'CPUExecutionProvider' in ort.get_available_providers()

    def unavailable_error(self) -> RuntimeError:
        return RuntimeError(
            'CPU tensor conversion requires CPUExecutionProvider')

    def allocate(
        self,
        metadata: TensorMetadata,
        device_id: int,
        stream: Optional[int],
    ) -> object:
        del stream
        if device_id != 0:
            raise ValueError('CPU tensors require device_id 0')
        return array('B', bytes(metadata.byte_count))

    def view(
        self,
        message: ExperimentalTensor,
        metadata: TensorMetadata,
        stream: Optional[int],
        output: bool,
    ) -> OrtTensorView:
        del output
        if stream is not None:
            raise ValueError('CPU tensor conversion does not accept a stream')
        storage = np.ndarray(
            metadata.shape,
            dtype=metadata.numpy_dtype,
            buffer=message.data,
            offset=metadata.byte_offset,
            order='C',
        )
        if metadata.element_type == 16:
            value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(
                storage, metadata.element_type)
        else:
            value = ort.OrtValue.ortvalue_from_numpy(storage)
        return OrtTensorView(value, message, storage)
