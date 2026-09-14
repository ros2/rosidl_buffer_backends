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

"""Report the installed Torch version and device support."""
import importlib
import pathlib
import re
import sys

module = importlib.import_module(sys.argv[1])
expected = sys.argv[2]
require_cuda = sys.argv[3] == 'TRUE'
match = re.match(r'(\d+\.\d+\.\d+)', module.__version__)
if match is None:
    raise RuntimeError(f'Cannot identify the framework version: {module.__version__}')
release = match.group(1)
if expected.startswith('>='):
    minimum = expected[2:]
    lower = tuple(int(part) for part in minimum.split('.'))
    actual = tuple(int(part) for part in release.split('.'))
    if actual < lower:
        raise RuntimeError(f'Require >={minimum}, found {module.__version__}')
elif release != expected:
    raise RuntimeError(f'Require {expected}, found {module.__version__}')
variant = 'cpu' if module.version.cuda is None else 'cu' + module.version.cuda.replace('.', '')
if require_cuda and module.version.cuda is None:
    raise RuntimeError('CUDA-enabled Torch is required')
print(pathlib.Path(module.__file__).resolve().parent.parent)
print(variant)
print(release)
