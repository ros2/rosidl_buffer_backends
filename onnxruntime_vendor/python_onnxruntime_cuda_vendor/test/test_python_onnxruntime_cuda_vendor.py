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

import ctypes
import importlib.metadata
import os
from pathlib import Path
import runpy

import onnxruntime as ort


def test_vendored_cuda_runtime():
    expected_package = os.environ['PYTHON_ONNXRUNTIME_VENDOR_PACKAGE']
    expected_version = os.environ['PYTHON_ONNXRUNTIME_VENDOR_VERSION']
    expected_variant = os.environ['PYTHON_ONNXRUNTIME_EXPECTED_VARIANT']
    core_dir = Path(
        os.environ['PYTHON_ONNXRUNTIME_VENDOR_CORE_LIBDIR']).resolve()
    provider_library = Path(
        os.environ['PYTHON_ONNXRUNTIME_VENDOR_PROVIDER_LIBRARY']).resolve()
    capi_dir = Path(ort.__file__).parent / 'capi'

    distribution = importlib.metadata.distribution(expected_package)
    assert distribution.metadata['Name'].lower() == expected_package
    assert distribution.version == expected_version
    assert ort.__version__ == expected_version

    build_info = runpy.run_path(str(capi_dir / 'build_and_package_info.py'))
    assert build_info['package_name'] == expected_package
    assert build_info['__version__'] == expected_version
    cuda_version = build_info.get('cuda_version')
    assert cuda_version
    assert cuda_version.split('.')[0] == expected_variant.removeprefix('cuda')

    providers = ort.get_available_providers()
    assert 'CPUExecutionProvider' in providers
    assert 'CUDAExecutionProvider' in providers

    native_libraries = []
    for pattern in (
        'libonnxruntime.so*',
        'libonnxruntime_providers_shared.so*',
        'libonnxruntime_providers_cuda.so*',
    ):
        native_libraries.extend(capi_dir.glob(pattern))
    assert native_libraries
    assert all(path.is_symlink() for path in native_libraries)
    core_libraries = [
        path for path in native_libraries
        if 'providers_cuda' not in path.name
    ]
    assert all(path.resolve().is_relative_to(core_dir) for path in core_libraries)
    cuda_libraries = [
        path for path in native_libraries
        if 'providers_cuda' in path.name
    ]
    assert len(cuda_libraries) == 1
    assert cuda_libraries[0].resolve() == provider_library
    assert not list(capi_dir.glob('libonnxruntime_providers_tensorrt.so*'))

    ctypes.CDLL(str(capi_dir / 'libonnxruntime.so'))
    mapped_libraries = {
        Path(line.rsplit(maxsplit=1)[-1]).resolve()
        for line in Path('/proc/self/maps').read_text().splitlines()
        if '/libonnxruntime.so' in line
    }
    assert mapped_libraries
    assert all(path.is_relative_to(core_dir) for path in mapped_libraries)
