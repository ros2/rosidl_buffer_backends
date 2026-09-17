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

"""Extract the native SDK and its dependencies without changing wheel RPATHs."""

from email.parser import Parser
from pathlib import Path
import sys
from zipfile import ZipFile


def stage(wheels, root, version):
    torch_wheels = list(wheels.glob('torch-*.whl'))
    if len(torch_wheels) != 1:
        raise RuntimeError(f'Expected one Torch wheel, found {len(torch_wheels)}')
    for wheel in wheels.glob('*.whl'):
        with ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata_path = next(n for n in names if n.endswith('.dist-info/METADATA'))
            metadata = Parser().parsestr(archive.read(metadata_path).decode())
            is_torch = metadata['Name'] == 'torch'
            if is_torch and metadata['Version'] != version:
                raise RuntimeError(f"Expected Torch {version}, found {metadata['Version']}")
            if not is_torch and not metadata['Name'].startswith('nvidia-'):
                continue
            for name in names:
                path = Path(name)
                if path.is_absolute() or '..' in path.parts:
                    raise RuntimeError(f'Invalid wheel member: {name}')
                if name.startswith(('torch/include/', 'torch/lib/', 'torch/share/', 'nvidia/')):
                    archive.extract(name, root)
                elif '.dist-info/licenses/' in name or name.endswith('.dist-info/LICENSE'):
                    archive.extract(name, root)
    sdk = root / 'torch'
    for name in ['share/cmake/Torch/TorchConfig.cmake', 'include/ATen/ATen.h', 'lib/libtorch.so']:
        if not (sdk / name).is_file():
            raise RuntimeError(f'Torch wheel is missing {name}')
    (sdk / 'build-version').write_text(version + '\n')
    for pattern in [
        'nvidia/*/lib/libcufile_rdma*',
        'nvidia/nvshmem/lib/nvshmem_transport_ib*',
        'nvidia/nvshmem/lib/nvshmem_transport_libfabric*',
        'nvidia/nvshmem/lib/nvshmem_transport_ucx*',
        'nvidia/nvshmem/lib/nvshmem_bootstrap_mpi*',
        'nvidia/nvshmem/lib/nvshmem_bootstrap_pmix*',
        'nvidia/nvshmem/lib/nvshmem_bootstrap_shmem*',
    ]:
        for path in root.glob(pattern):
            path.unlink()


if __name__ == '__main__':
    stage(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
