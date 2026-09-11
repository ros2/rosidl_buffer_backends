# Clean ONNX Runtime Debian build and staged plugin validation

This directory models a ROS build-farm binary job and a later user install in
two independent Docker environments.

## GPU-less build-farm container

`Dockerfile.buildfarm` starts from `ros:rolling-ros-base-resolute` and asserts
that CUDA, an NVIDIA driver/device, ONNX Runtime, and Python ONNX Runtime are
absent. It adds only generic packaging tools. `build-debians.sh` then installs
the complete source dependency closure from `package.xml` through rosdep and
builds 15 ROS Debians with Bloom and unmodified generated rules:

- `tensor_msgs`
- `cuda_buffer_backend_msgs`, `cuda_buffer`, `cuda_buffer_backend`,
  `cuda_buffer_py`
- `onnxruntime_core_vendor`, `onnxruntime_cuda_vendor`
- `python_onnxruntime_vendor`, `python_onnxruntime_cuda_vendor`
- `onnxruntime_conversions`, `onnxruntime_conversions_cpu`,
  `onnxruntime_conversions_cuda`
- `onnxruntime_conversions_py`, `onnxruntime_conversions_py_cpu`,
  `onnxruntime_conversions_py_cuda`

CUDA and cuDNN use the public `nvidia-cuda` and `nvidia-cudnn` rosdep keys.
The local rosdep YAML contains only release-name mappings for ROS packages in
this unreleased source repository and a Resolute name for the package used in
a declared conflict. No GPU or host driver is passed into the build.

The CPU-only providers are built but not installed into the build root because
they conflict with the CUDA-capable providers. The conversion cores depend on
the CUDA-capable providers from the start, so their CPU plugins exercise the
same process-wide framework ABI that the later CUDA plugins use. All four C++
and Python providers use ONNX Runtime 1.26.0; the CUDA providers require CUDA
12.

## Pristine consumer and unshipped tests

`run-consumer-test.sh` starts from another untouched ROS image and again
asserts that no CUDA or ONNX Runtime input is present. It adds the generated
Debian repository and installs only the C++/Python cores and CPU plugins.
Their declared dependencies pull the framework providers, CUDA user-space
libraries, and cuDNN from the generated/public repositories; no conversion
CUDA plugin is installed yet.

Production Debians intentionally contain no unit or launch test executables.
`test-installed-debians.sh` therefore builds
`installed_tests/onnxruntime_conversions_installed_tests` from the mounted
source checkout against the installed Debians, then invokes its binaries and
launch file manually. It:

1. checks CPU-only plugin discovery and runs the C++ and Python CPU tests;
2. snapshots the installed C++ and Python core files;
3. installs only the independently built C++ and Python CUDA plugins;
4. verifies the installed files are byte-for-byte unchanged;
5. runs CUDA discovery, C++/Python CUDA tests, the C++ launch test, and the
   Python Fast RTPS process-boundary test on the GPU; and
6. selects CPU, CUDA, then CPU again within one Python process.

Plugin discovery happens once per process. Post-install checks start new
processes; already-running applications must be restarted after a new plugin
Debian is installed.

## Run and outputs

From the repository root:

```bash
bash docker/resolute/debian/test-pipeline.sh
```

The build and pristine consumer need public package/download access. The GPU
consumer additionally needs NVIDIA Container Toolkit and a compatible host
driver.

Generated Debians, apt metadata, and dependency packages are written under
`artifacts/` and ignored by Git. The complete installed-test log and its
summary are written under `reports/`.

## Architecture scope

The pipeline validates CUDA 12 on amd64. Generic arm64 is not claimed by this
report.
