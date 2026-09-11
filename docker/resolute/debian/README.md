# Clean Debian build and staged CPU-to-CUDA validation

This directory models a ROS build-farm build and a later user installation in
two isolated Docker environments.

## Build-farm container

`Dockerfile.buildfarm` starts from `ros:rolling-ros-base-resolute` and asserts
that nvcc, CUDA runtime libraries, LibTorch, Python Torch, the NVIDIA driver,
and GPU devices are absent. It adds only generic Debian packaging tools.
`build-debians.sh` then installs every project dependency from `package.xml`
through rosdep.

CUDA uses the upstream `nvidia-cuda` rosdep key. No private CUDA repository or
local external-dependency mapping is used, and the build receives no host GPU
or driver. The local rosdep file contains only mappings for the unreleased ROS
packages in this repository. Bloom and `dpkg-buildpackage` generate and install
these 15 ROS Debians in dependency order:

- `ros-rolling-tensor-msgs`
- `ros-rolling-cuda-buffer-backend-msgs`
- `ros-rolling-cuda-buffer`
- `ros-rolling-cuda-buffer-backend`
- `ros-rolling-cuda-buffer-py`
- `ros-rolling-libtorch-vendor`
- `ros-rolling-python3-torch-vendor`
- `ros-rolling-torch-conversions`
- `ros-rolling-torch-conversions-cpu`
- `ros-rolling-torch-conversions-py`
- `ros-rolling-torch-conversions-py-cpu`
- `ros-rolling-libtorch-cuda-vendor`
- `ros-rolling-python3-torch-cuda-vendor`
- `ros-rolling-torch-conversions-cuda`
- `ros-rolling-torch-conversions-py-cuda`

The CPU providers install official CPU LibTorch and Python Torch 2.9.1
distributions. The CUDA providers select CUDA 12-compatible Torch 2.9.1
distributions from the toolkit installed by rosdep. CPU conversion packages
are built before either CUDA provider is installed into the build root, so
CUDA environment hooks cannot influence their compile or link commands. The
Python CUDA provider removes optional RDMA transport DSOs whose system
dependencies have no Resolute rosdep keys and pre-strips upstream wheel ELFs.
The C++ providers omit the archive's Python-ABI and test-only libraries; Python
bindings come exclusively from the Python providers. The pipeline does not
edit Bloom-generated `debian/rules`.

Dependency Debians fetched by rosdep are retained beside the generated ROS
Debians. `Packages`, `Packages.gz`, `MANIFEST.tsv`, and `SHA256SUMS` turn that
directory into a self-contained offline apt repository.

## Pristine consumer and external tests

`run-offline-test.sh` starts in a second untouched ROS image and again asserts
that CUDA and Torch are absent. It disables all external apt sources and
installs only these conversion packages initially:

- `torch_conversions` and `torch_conversions_cpu`
- `torch_conversions_py` and `torch_conversions_py_cpu`

Their CPU providers use CPU-only LibTorch/Python Torch distributions, and
no CUDA package or conversion plugin is installed or discoverable in this
stage. Installing the CUDA plugins later brings matching CUDA provider overlays
into the ROS prefix; new processes use the overlays for both CPU and CUDA.
The CPU provider files remain installed and byte-identical throughout this
upgrade.

Production Debians do not ship the launch tests. Therefore
`test-installed-debians.sh` restores the build container’s rosdep cache, builds
`installed_tests/torch_conversions_installed_tests` from the mounted repository
source, and invokes the resulting tests manually. It performs:

1. CPU runtime discovery and round trip;
2. 11 C++ CPU unit tests and 24 collected Python CPU tests;
3. four C++ and one Python Fast RTPS launch tests on CPU;
4. installation of only the C++ and Python CUDA plugin Debians;
5. SHA-256 equality checks for the installed C++ and Python core files;
6. four C++ and seven Python CUDA tests;
7. the same five launch tests on CUDA; and
8. Python CPU-to-CUDA-to-CPU selection in one process.

Plugin discovery occurs once per process, so the post-install tests use new
processes. An already-running process retains its loaded provider and plugin
registry.

## Run and artifacts

From the repository root:

```bash
bash docker/resolute/debian/test-pipeline.sh
```

The build phase needs ordinary package-download network access but no GPU. The
consumer is offline and needs an NVIDIA GPU, compatible host driver, and
NVIDIA Container Toolkit.

Generated Debians and apt metadata are written under `artifacts/` and ignored
by Git. The complete test log and summarized result are written under
`reports/`.

## Architecture scope

The downloaded CPU and CUDA LibTorch/Python Torch distributions are validated
on amd64. On arm64,
the CUDA providers accept only NVIDIA Tegra and reuse compatible JetPack
distributions; generic CUDA arm64 is intentionally rejected and is not
validated by this pipeline.
