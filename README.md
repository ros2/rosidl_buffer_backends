# rosidl_buffer_backends

Vendor-specific storage backends for `rosidl::Buffer`, enabling ROS 2
publishers and subscribers to share accelerator-accessible memory without
copying when supported, with automatic fallback to the default CPU path.

## Packages

- [`cuda_buffer_backend`](cuda_buffer_backend/README.md) -- CUDA buffer backend
  packages for zero-copy GPU memory sharing using CUDA VMM IPC.
- [`qc_buffer_backend`](qc_buffer_backend/README.md) -- Qualcomm buffer backend
  packages for HTP-CPU zero-copy sharing using dma-buf memory.
- **libtorch_vendor** -- Vendor package that downloads and installs the
  pre-built LibTorch C++ distribution.
- **tensor_msgs** -- DLPack-aligned `ExperimentalTensor.msg` definition.
- **torch_conversions** -- Header-only helper library that converts between
  `tensor_msgs/ExperimentalTensor` and `at::Tensor` and exposes DLPack import /
  export. Replaces the older `torch_buffer_backend` plugin approach with a
  plain message + bridge library that rides on top of whichever
  `rosidl::Buffer` backend is registered (CUDA when available, CPU
  otherwise).

## Deb build status

### ROS 2 Lyrical (Ubuntu Resolute)

| Package | Source deb | Binary deb (amd64) | Binary deb (arm64) |
| ------- | ---------- | ------------------ | ------------------ |
| cuda_buffer | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__cuda_buffer__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__cuda_buffer__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__cuda_buffer__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__cuda_buffer__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__cuda_buffer__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__cuda_buffer__ubuntu_resolute_arm64__binary/) |
| cuda_buffer_backend | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__cuda_buffer_backend__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__cuda_buffer_backend__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__cuda_buffer_backend__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__cuda_buffer_backend__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__cuda_buffer_backend__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__cuda_buffer_backend__ubuntu_resolute_arm64__binary/) |
| cuda_buffer_backend_msgs | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__cuda_buffer_backend_msgs__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__cuda_buffer_backend_msgs__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__cuda_buffer_backend_msgs__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__cuda_buffer_backend_msgs__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__cuda_buffer_backend_msgs__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__cuda_buffer_backend_msgs__ubuntu_resolute_arm64__binary/) |
| libtorch_vendor | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__libtorch_vendor__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__libtorch_vendor__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__libtorch_vendor__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__libtorch_vendor__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__libtorch_vendor__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__libtorch_vendor__ubuntu_resolute_arm64__binary/) |
| tensor_msgs | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__tensor_msgs__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__tensor_msgs__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__tensor_msgs__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__tensor_msgs__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__tensor_msgs__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__tensor_msgs__ubuntu_resolute_arm64__binary/) |
| torch_conversions | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__torch_conversions__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__torch_conversions__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__torch_conversions__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__torch_conversions__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__torch_conversions__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__torch_conversions__ubuntu_resolute_arm64__binary/) |

### ROS 2 Rolling (Ubuntu Resolute)

| Package | Source deb | Binary deb (amd64) | Binary deb (arm64) |
| ------- | ---------- | ------------------ | ------------------ |
| cuda_buffer | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__cuda_buffer__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__cuda_buffer__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__cuda_buffer__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__cuda_buffer__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__cuda_buffer__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__cuda_buffer__ubuntu_resolute_arm64__binary/) |
| cuda_buffer_backend | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__cuda_buffer_backend__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__cuda_buffer_backend__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__cuda_buffer_backend__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__cuda_buffer_backend__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__cuda_buffer_backend__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__cuda_buffer_backend__ubuntu_resolute_arm64__binary/) |
| cuda_buffer_backend_msgs | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__cuda_buffer_backend_msgs__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__cuda_buffer_backend_msgs__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__cuda_buffer_backend_msgs__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__cuda_buffer_backend_msgs__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__cuda_buffer_backend_msgs__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__cuda_buffer_backend_msgs__ubuntu_resolute_arm64__binary/) |
| libtorch_vendor | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__libtorch_vendor__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__libtorch_vendor__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__libtorch_vendor__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__libtorch_vendor__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__libtorch_vendor__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__libtorch_vendor__ubuntu_resolute_arm64__binary/) |
| tensor_msgs | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__tensor_msgs__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__tensor_msgs__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__tensor_msgs__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__tensor_msgs__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__tensor_msgs__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__tensor_msgs__ubuntu_resolute_arm64__binary/) |
| torch_conversions | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__torch_conversions__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__torch_conversions__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__torch_conversions__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__torch_conversions__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__torch_conversions__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__torch_conversions__ubuntu_resolute_arm64__binary/) |

## Prerequisites

- A ROS 2 Rolling development environment. See the upstream
  [Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
  guide for the canonical source-build flow, or use the pixi workflow
  shipped by the [`ros2/ros2`](https://github.com/ros2/ros2) meta-repo.
- Each backend depends on its vendor-specific platform support. Refer to the
  backend documentation for detailed prerequisites, setup, build, test, and
  usage instructions:

- [`cuda_buffer_backend/README.md`](cuda_buffer_backend/README.md)
- [`qc_buffer_backend/README.md`](qc_buffer_backend/README.md)
- [`torch_conversions/README.md`](torch_conversions/README.md)

## License

Apache-2.0
