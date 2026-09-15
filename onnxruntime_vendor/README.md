# ONNX Runtime providers

Source builds accept stable **ONNX Runtime >=1.27.0** that pass the version,
provider and API checks. The fallback version is **1.29.0**.

| Vendor | Accepted provider | Fallback |
| --- | --- | --- |
| `onnxruntime_core_vendor` | CPU or CUDA 12/13 C++ SDK | CPU SDK |
| `onnxruntime_cuda_vendor` | C++ CUDA provider matching the toolkit major | `gpu_cuda12` or `gpu_cuda13` SDK |
| `python_onnxruntime_vendor` | CPU or CUDA 12/13 Python distribution | `onnxruntime` wheel |
| `python_onnxruntime_cuda_vendor` | Python CUDA provider matching the toolkit major | `onnxruntime-gpu` wheel |

## Source builds

Set `onnxruntime_ROOT` or `CMAKE_PREFIX_PATH` to select a C++ SDK. Reuse checks
the headers, runtime and a compiled CPU tensor operation. Headers must expose
C API >=27, and the runtime must provide the requested API. For 1.x releases,
the header API version must equal the runtime minor version.

Select a Python interpreter with `-DPython3_EXECUTABLE=/path/to/venv/bin/python`.
Its Python major/minor version must match the ROS Python bindings. Python reuse
checks the import, release, `OrtValue.from_dlpack`, provider build and a CPU
tensor operation. C++ consumers require a separate SDK with headers.

If discovery or compatibility checks fail, the vendor installs the fallback
under `opt/<vendor-package>` in the ROS prefix. `-DFORCE_BUILD_VENDOR_PKG=ON`
selects the fallback directly. CPU and CUDA vendors must select the same release
within each C++ or Python stack. A fallback whose release differs from the
selected CPU/core provider causes a configuration error. Configure matching
providers or rebuild both vendors with `-DFORCE_BUILD_VENDOR_PKG=ON`.

Native conversions, plugins and applications must use the framework release
selected when the conversion core was built. Rebuild dependent C++ packages
when changing that release. Use a fresh build directory when changing providers
or toolkits. Reused paths must remain available to applications. Probe results
are saved as `onnxruntime_reuse.log` or `python_reuse.log` in the vendor's build
directory.

## CUDA

Source builds require CUDA **>=12,<14**. CUDA providers must match the selected
toolkit's major. Set `-DCUDAToolkit_ROOT=/usr/local/cuda-13.1` in a fresh build to
select a toolkit. A configured CUDA compiler can take precedence; check the
version and path in CMake's output. When supplying the toolkit yourself, use
`--skip-keys cuda-toolkit-13-1` with `rosdep install`.

CUDA 13 Python wheels come from PyPI. CUDA 12 wheels come from the
[ONNX Runtime CUDA 12 feed](https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/).

The C++ CUDA vendor supplies cuDNN for the selected CUDA major; the Python CUDA
vendor depends on it. Set `CUDNN_ROOT` to select an existing cuDNN installation.
Reuse checks `cudnnGetVersion()` and `cudnnGetCudartVersion()`.

| Toolkit | Accepted cuDNN | Fallback |
| --- | --- | --- |
| CUDA 12 | >=9.10.2,<10, built for CUDA 12 | `nvidia-cudnn-cu12==9.10.2.21` |
| CUDA 13 | >=9.24.0,<10, built for CUDA 13 | `nvidia-cudnn-cu13==9.24.0.43` |

The toolkit supplies cuBLAS. CUDA 12 toolkits older than 12.8 also use the
vendor's `nvidia-cuda-runtime-cu12==12.8.90` fallback. ROS environment hooks add
these runtime libraries to the library search path. GPU execution requires a
compatible NVIDIA driver.

## Debian installation

The Ubuntu Resolute dependency is `cuda-toolkit-13-1` (>=13.1,<13.2). APT checks
installed Debian package names and versions.

Vendor Debians contain the fallback framework under the ROS prefix. Existing
pip or Conda installations can coexist with them. Sourcing ROS selects the
vendor's Python paths and native libraries; CUDA providers take precedence over
CPU providers. Start a new process after changing providers. For source reuse
of another installation, build against that installation in a separate environment.
Release builds require `-DFORCE_BUILD_VENDOR_PKG=ON`.

## Platform limits

CPU SDK fallbacks target Linux amd64 and arm64; CUDA SDK fallbacks target Linux
amd64. Cross compilation selects the fallback and skips the executable reuse
probes. Runtime testing covers Linux amd64 with CUDA 12.6 and 13.1.

Upstream requirements: [ONNX Runtime CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
