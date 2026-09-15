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

"""Check Python Torch version, device support and tensor operations."""
import importlib
import pathlib
import re
import sys

module = importlib.import_module(sys.argv[1])
expected = sys.argv[2]
require_cuda = sys.argv[3] == 'TRUE'
cuda_version = sys.argv[4] if len(sys.argv) > 4 else ''
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
variant = 'cpu' if module.version.cuda is None else 'cu' + module.version.cuda.replace('.', '')
if variant != 'cpu' and not re.fullmatch(r'cu(?:12|13)[0-9]+', variant):
    raise RuntimeError(f'Unsupported Torch build: {variant}')
if require_cuda:
    if module.version.cuda is None:
        raise RuntimeError('CUDA-enabled Torch is required')
    toolkit = tuple(int(part) for part in cuda_version.split('.')[:2])
    framework = tuple(int(part) for part in module.version.cuda.split('.')[:2])
    reason = ''
    if toolkit < (12, 6) or toolkit >= (14, 0):
        reason = 'selected toolkit is outside the supported range >=12.6,<14'
    elif framework[0] != toolkit[0]:
        reason = 'CUDA major versions differ'
    elif toolkit < framework:
        reason = 'the Torch CUDA build is newer than the selected toolkit'
    if reason:
        raise RuntimeError(
            f'Rejected Torch {variant} (built with CUDA {module.version.cuda}): '
            f'selected CUDA toolkit is {cuda_version}; {reason}. '
            'Require the same CUDA major and an older or equal Torch CUDA build minor. '
            'Select a compatible toolkit with CUDAToolkit_ROOT or a compatible '
            'Python Torch installation with Python3_EXECUTABLE.')
if module.ones((2, 3)).sum().item() != 6:
    raise RuntimeError('Torch CPU operation failed')
print(pathlib.Path(module.__file__).resolve().parent.parent)
print(variant)
print(release)
