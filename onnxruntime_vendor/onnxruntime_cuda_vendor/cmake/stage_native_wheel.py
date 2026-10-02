# Copyright 2026 NVIDIA Corporation
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

"""Add CUDA wheel libraries to the matching ONNX Runtime release SDK."""

import ast
from email.parser import Parser
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile


def stage(wheels, root, version, cuda_major):
    candidates = list(wheels.glob('onnxruntime_gpu-*.whl'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one ONNX Runtime GPU wheel, found {len(candidates)}')
    with ZipFile(candidates[0]) as archive:
        metadata_path = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        if metadata['Version'] != version:
            raise RuntimeError(f"Expected ONNX Runtime {version}, found {metadata['Version']}")
        info = {}
        tree = ast.parse(archive.read('onnxruntime/capi/build_and_package_info.py'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                info[node.targets[0].id] = ast.literal_eval(node.value)
        if str(info.get('cuda_version', '')).split('.')[0] != cuda_major:
            raise RuntimeError(f"Expected CUDA {cuda_major}, found {info.get('cuda_version')}")
        libraries = [f'libonnxruntime.so.{version}',
                     'libonnxruntime_providers_shared.so', 'libonnxruntime_providers_cuda.so']
        for name in libraries:
            archive.getinfo('onnxruntime/capi/' + name)
        for path in (root / 'lib').glob('libonnxruntime*.so*'):
            path.unlink()
        for name in libraries:
            with archive.open('onnxruntime/capi/' + name) as src:
                with (root / 'lib' / name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        (root / 'lib' / 'libonnxruntime.so.1').symlink_to(f'libonnxruntime.so.{version}')
        (root / 'lib' / 'libonnxruntime.so').symlink_to('libonnxruntime.so.1')


if __name__ == '__main__':
    stage(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4])
