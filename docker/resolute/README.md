# ROS 2 Rolling on Ubuntu Resolute: PR #11

This directory contains reproducible Docker environments for the Torch
conversion implementation on top of
[`ros2/rosidl_buffer_backends#11`](https://github.com/ros2/rosidl_buffer_backends/pull/11).

All project dependencies are declared in `package.xml` and installed through
rosdep. CUDA resolves through the public `nvidia-cuda` key; no NVIDIA apt
repository, preinstalled CUDA tree, Python Torch installation, or private
external-dependency rule is used.

## Build and test the current source tree

From the repository root:

```bash
bash docker/resolute/build.sh
```

The image builds the C++ and Python cores plus both CPU and CUDA plugins. It
then builds the source-only test harness under `debian/installed_tests/` and
runs the CPU unit and launch tests during the GPU-less image build.

Run the resulting image again without or with a GPU:

```bash
docker run --rm rosidl-buffer-backends:pr-11-resolute
docker run --rm --gpus all rosidl-buffer-backends:pr-11-resolute
```

The first command exercises the CPU implementation and lets CUDA-specific
tests remain disabled. The second discovers the installed CUDA plugins and
runs the C++ and Python CUDA tests as well.

## Build Debians and validate staged plugin installation

The `debian/` pipeline is the release-like acceptance test:

```bash
bash docker/resolute/debian/test-pipeline.sh
```

A GPU-less build container starts without CUDA, LibTorch, or Python Torch,
installs the complete dependency closure through rosdep, and builds all 15 ROS
Debians with unmodified Bloom-generated rules. A second pristine container
disables external apt sources, installs the cores and CPU plugins first, builds
the unshipped tests from source, and runs them manually without CUDA packages.
It then installs only the CUDA conversion plugins and their CUDA providers,
proves that neither core file changed, and reruns the full C++/Python unit and
launch matrix on the GPU.

See [`debian/README.md`](debian/README.md) for the exact package and test flow,
and [`../../torch_conversions/README.md`](../../torch_conversions/README.md)
for the ABI/provider boundary.

The final GPU run requires NVIDIA Container Toolkit and a compatible NVIDIA
driver on the host.
