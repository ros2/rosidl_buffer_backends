# ONNX Runtime Debian validation

Date: 2026-09-11 (UTC)

## Result

PASS — all 15 in-scope repository packages were generated as binary Debians
with standard Bloom/debhelper rules in a clean ROS 2 Rolling/Ubuntu Resolute
Docker builder. The builder started without a GPU device, NVIDIA driver, CUDA
toolkit, ONNX Runtime, or Torch. `rosdep` resolved the build and runtime keys
from `package.xml`; the local rosdep file represents the future rosdistro
entries for unreleased packages.

The artifact repository contains 80 generated and cached Debians and every
entry passes `sha256sum -c SHA256SUMS`. The 15 project Debians are the five
message/CUDA-buffer foundations, four mutually exclusive ONNX Runtime
providers, and six ONNX Runtime conversion core/plugin packages. The two
CPU-only providers were built as standalone artifacts and intentionally not
installed in the builder alongside the conflicting CUDA-capable providers.

All four provider packages supply ONNX Runtime 1.26.0. The installed-package
test explicitly verified the C++ and Python provider versions and confirmed
that Python exposes `OrtValue.from_dlpack`, `__dlpack__`, and
`__dlpack_device__`.

Ubuntu Resolute supplies CUDA 12.4, while the official ONNX Runtime CUDA 12
binary requires the CUDA 12.8 runtime ABI. The exclusive
`onnxruntime_cuda_vendor` Debian therefore includes NVIDIA's pinned,
SHA-256-verified `libcudart.so.12` 12.8 compatibility runtime. CUDA headers and
the rest of the toolkit still come from the manifest's `nvidia-cuda` rosdep
key. The provider package also carries the NVIDIA runtime license.

## Installed-package tests

The production conversion Debians contain no test or launch assets. After
installing the Debians, the consumer copied a dedicated source-only test
package to `/tmp`, compiled the original C++ tests and ROS nodes against the
installed packages, generated the launch file from the original source, and
launched it manually.

### Stage 1: core and CPU plugins only

- C++ CPU unit tests: 20/20 passed.
- Python CPU unit tests: 45/45 passed.
- Both CUDA conversion plugin Debians were confirmed absent.
- Runtime discovery reported `available=cpu, default=cpu`.
- The C++ and Python ONNX Runtime providers both reported version 1.26.0.

### Stage 2: install CUDA plugins without replacing cores or providers

- Installed only `ros-rolling-onnxruntime-conversions-cuda` and
  `ros-rolling-onnxruntime-conversions-py-cuda` (plus their generated CUDA
  buffer dependencies) from the local artifact repository.
- APT was pinned to the exact locally generated `0.1.2-0resolute` version so a
  published higher-revision package could not be substituted.
- SHA-256 checks confirmed that the C++ and Python conversion cores and both
  ONNX Runtime provider binaries were byte-identical before and after plugin
  installation.
- GPU used: NVIDIA GeForce RTX 4080, driver 610.57.04.

### Stage 3: installed CUDA plugins

- C++ CUDA unit tests: 9/9 passed.
- Python CUDA unit tests: 13/13 passed.
- C++ Fast DDS inter-process CUDA inference launch test: passed, including
  clean process shutdown.
- Python Fast DDS inter-process CUDA inference test: 1/1 passed.
- A single Python process selected CPU, then CUDA, then CPU again successfully.
- Final marker: `staged-debian-installed-source-tests=pass`.

## Production package-content audit

`dpkg-deb -c` confirmed that none of these six production Debians ships a
`test`, `tests`, or launch-test path:

- `ros-rolling-onnxruntime-conversions`
- `ros-rolling-onnxruntime-conversions-cpu`
- `ros-rolling-onnxruntime-conversions-cuda`
- `ros-rolling-onnxruntime-conversions-py`
- `ros-rolling-onnxruntime-conversions-py-cpu`
- `ros-rolling-onnxruntime-conversions-py-cuda`

The complete installed-test output is in
[`onnxruntime-debian-installed-tests.log`](onnxruntime-debian-installed-tests.log).

## Provider packaging audit

- `ros-rolling-onnxruntime-core-vendor` and
  `ros-rolling-python-onnxruntime-vendor` contain the CPU-only 1.26.0
  distributions and have no CUDA or NVIDIA package dependency.
- `ros-rolling-onnxruntime-cuda-vendor` and
  `ros-rolling-python-onnxruntime-cuda-vendor` contain the CUDA 12-compatible
  1.26.0 distributions and conflict with their CPU-only counterparts.
- All provider downloads are version-pinned and SHA-256 verified.
- CUDA manifests consistently constrain the toolkit to major version 12.

Ubuntu Resolute's public ONNX Runtime 1.23 packages were not used for the
conversion packages. The packaged Python API does not expose the DLPack entry
points required by the zero-copy implementation, and its available execution
providers are CPU-only. Mixing that C++ provider with the CUDA 1.26 provider
would also introduce two incompatible ONNX Runtime ABIs into one process.

## Final source audit

- All 10 ONNX Runtime package manifests use repository version 0.1.2.
- Provider payload versions are consistently ONNX Runtime 1.26.0.
- ROS linters passed: `ament_flake8`, `ament_pep257`, `ament_cpplint`,
  `ament_uncrustify`, `ament_xmllint`, `ament_lint_cmake`, and
  `ament_copyright`.
- Shell syntax, Python syntax, manifest parsing, and `git diff --check` passed.
- No generated Python cache files or obsolete DLPack conversion package remain.
