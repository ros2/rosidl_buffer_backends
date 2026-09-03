#!/usr/bin/env python3

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

import importlib.metadata
from pathlib import Path
import runpy
import sys


python_dir = Path(sys.argv[1])
core_dir = Path(sys.argv[2])
relative_core_dir = Path(sys.argv[3])
expected_package = sys.argv[4]
expected_version = sys.argv[5]
capi_dir = python_dir / 'onnxruntime' / 'capi'

distributions = {
    distribution.metadata['Name'].lower(): distribution
    for distribution in importlib.metadata.distributions(path=[python_dir])
}
distribution = distributions[expected_package]
assert distribution.version == expected_version
assert distribution.metadata['Name'].lower() == expected_package

build_info = runpy.run_path(str(capi_dir / 'build_and_package_info.py'))
assert build_info['package_name'] == expected_package
assert build_info['__version__'] == expected_version

cuda_version = build_info.get('cuda_version')
assert not cuda_version

native_patterns = (
    'libonnxruntime.so*',
    'libonnxruntime_providers_shared.so*',
    'libonnxruntime_providers_cuda.so*',
    'libonnxruntime_providers_tensorrt.so*',
)
for pattern in native_patterns:
    for path in capi_dir.glob(pattern):
        path.unlink()

link_patterns = [
    'libonnxruntime.so*',
    'libonnxruntime_providers_shared.so*',
]

for pattern in link_patterns:
    sources = list(core_dir.glob(pattern))
    assert sources
    for source in sources:
        (capi_dir / source.name).symlink_to(relative_core_dir / source.name)

for pattern in native_patterns:
    assert all(path.is_symlink() for path in capi_dir.glob(pattern))
