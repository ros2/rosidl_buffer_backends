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

"""Check Python ONNX Runtime version, device support and tensor operations."""
import importlib
import pathlib
import re
import sys

import numpy as np

module = importlib.import_module(sys.argv[1])
expected = sys.argv[2]
require_cuda = sys.argv[3] == 'TRUE'
cuda_major = sys.argv[4] if len(sys.argv) > 4 else ''
release = module.__version__.split('+')[0]
if not re.fullmatch(r'\d+\.\d+\.\d+', release):
    raise RuntimeError(f'Require a stable framework release, found {module.__version__}')
if expected.startswith('>='):
    minimum = expected[2:]
    lower = tuple(int(part) for part in minimum.split('.'))
    actual = tuple(int(part) for part in release.split('.'))
    if actual < lower:
        raise RuntimeError(f'Require >={minimum}, found {module.__version__}')
elif release != expected:
    raise RuntimeError(f'Require {expected}, found {module.__version__}')
variant = 'cuda' if 'CUDAExecutionProvider' in module.get_available_providers() else 'cpu'
if variant == 'cuda':
    provider = pathlib.Path(module.__file__).parent / 'capi/libonnxruntime_providers_cuda.so'
    runtime_majors = re.findall(rb'libcudart\.so\.(12|13)\x00', provider.read_bytes())
    if not runtime_majors or (require_cuda and cuda_major.encode() not in runtime_majors):
        raise RuntimeError('ONNX Runtime and selected toolkit CUDA major differ')
if require_cuda and variant != 'cuda':
    raise RuntimeError('CUDAExecutionProvider is absent')
if not hasattr(module.OrtValue, 'from_dlpack'):
    raise RuntimeError('OrtValue.from_dlpack is absent')
value = module.OrtValue.ortvalue_from_numpy(np.ones((2, 3), dtype=np.float32))
if value.numpy().sum() != 6:
    raise RuntimeError('OrtValue operation failed')
print(pathlib.Path(module.__file__).resolve().parent.parent)
print(variant)
print(release)
