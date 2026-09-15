# Torch providers

Source builds accept stable **Torch/LibTorch >=2.9.0** that pass the version,
device and ABI checks. The fallback version is **2.14.0**.

| Vendor | Accepted provider | Fallback |
| --- | --- | --- |
| `libtorch_vendor` | CPU or CUDA LibTorch | CPU SDK |
| `libtorch_cuda_vendor` | CUDA LibTorch | CUDA SDK |
| `python3_torch_vendor` | CPU or CUDA Torch | CPU wheel |
| `python3_torch_cuda_vendor` | CUDA Torch | CUDA wheel |

## Source builds

C++ reuse checks `TorchConfig.cmake`, release and CUDA build metadata, and a
compiled CPU tensor operation. Native consumers require C++20, the libstdc++
C++11 ABI, and the framework release used to build the conversion core. CPU and
CUDA vendors must select the same release within each C++ or Python stack.
Rebuild dependent C++ packages when changing the framework release.

Select a C++ SDK with `Torch_DIR` or `CMAKE_PREFIX_PATH`. For an SDK in a pip
installation:

```bash
colcon build --cmake-args \
  -DTorch_DIR="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path + "/Torch")')"
```

Python reuse checks the import, release, CUDA build and a CPU tensor operation.
Select the interpreter with `-DPython3_EXECUTABLE=/path/to/venv/bin/python`.
Its Python major/minor version must match the ROS Python bindings.

If discovery or compatibility checks fail, the vendor installs the fallback
under `opt/<vendor-package>` in the ROS prefix. `-DFORCE_BUILD_VENDOR_PKG=ON`
selects the fallback directly. A fallback whose release differs from the
selected CPU/core provider causes a configuration error. Configure matching
providers or rebuild both vendors with `-DFORCE_BUILD_VENDOR_PKG=ON`.

Use a fresh build directory when changing providers or toolkits. Reused paths
must remain available to applications. Probe results are saved in the vendor's
build directory as `libtorch_reuse.log` or `python_reuse.log`.

C++ applications using CUDA Torch APIs must find `libtorch_cuda_vendor` before
importing `Torch` or `torch_conversions`. The CUDA vendor takes precedence over
the CPU vendor's SDK paths.

## CUDA

Source builds require CUDA **>=12.6,<14**. A Torch CUDA build must have the same
CUDA major and an older or equal minor compared with the selected toolkit.
For example, CUDA 13.1 accepts `cu130` and rejects `cu132`.

| Selected toolkit | Fallback variant |
| --- | --- |
| CUDA >=12.6,<13 | `cu126` |
| CUDA >=13,<13.2 | `cu130` |
| CUDA >=13.2,<14 | `cu132` |

`LIBTORCH_CUDA_VENDOR_VARIANT` and `PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT`
select a variant from this table. Explicit variants must satisfy the toolkit
rule and match any reused provider. Configuration errors identify the selected
toolkit and the rejected Torch CUDA build.

Set `-DCUDAToolkit_ROOT=/usr/local/cuda-13.1` in a fresh build to select a toolkit.
A configured CUDA compiler can take precedence; check the version and path in
CMake's output. When supplying the toolkit yourself, use
`--skip-keys cuda-toolkit` with `rosdep install`.

At runtime, Torch and ROS must load CUDA libraries that provide the symbols
required by both. Library search paths and libraries already loaded in the
process affect selection. GPU execution requires a compatible NVIDIA driver.

## Debian installation

The Ubuntu Resolute dependency is `cuda-toolkit` (>=13.1,<14). APT checks
installed Debian package names and versions. The metapackage selects the
repository's toolkit release. Generated Debians also depend on the CUDA runtime
packages required by their linked libraries.

Vendor Debians contain the fallback framework under the ROS prefix. Existing
pip or Conda installations can coexist with them. Sourcing ROS selects the
vendor's Python paths and native libraries; CUDA providers take precedence over
CPU providers. Start a new process after changing providers. For source reuse
of another installation, build against that installation in a separate environment.

Release builds require `-DFORCE_BUILD_VENDOR_PKG=ON`. The Python fallback uses
the system `python3-setuptools` package (>=77.0.3).

## Platform limits

Fallback SDKs and wheels target Linux amd64. Other architectures require
compatible existing providers. Cross compilation selects the fallback and skips
the executable reuse probes.

The C++ fallback contains the native libraries and headers. Python bindings are
provided by the Python vendors. The CUDA packages support local tensor operations;
distributed and multi-node workloads require additional transport libraries.

CUDA 13.1.115 with Resolute's glibc 2.43 can fail the `nvcc` compiler check on
conflicting `rsqrt`/`rsqrtf` exception specifications. Applications compiling
CUDA kernels require a compatible CUDA compiler and host toolchain.

Upstream distributions: [PyTorch wheel indexes](https://download.pytorch.org/whl/).
