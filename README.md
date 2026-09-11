# rosidl_buffer_backends

CUDA buffer backend implementation for `rosidl::Buffer`, enabling zero-copy
GPU memory sharing between ROS 2 publishers and subscribers, plus PyTorch and
ONNX Runtime conversion libraries built on the same buffer infrastructure.

## Packages

- **cuda_buffer** -- Core CUDA buffer library (VMM-backed IPC memory pool,
  host endpoint manager, ReadHandle/WriteHandle with CUDA event sync).
- **cuda_buffer_py** -- Python CUDA buffer allocation and scoped read/write
  handles for rclpy publishers and subscribers.
- **cuda_buffer_backend** -- BufferBackend plugin for CUDA IPC transport.
- **cuda_buffer_backend_msgs** -- ROS 2 message definitions for CUDA buffer
  descriptors.
- **libtorch_vendor** -- Official CPU LibTorch 2.9.1 provider.
- **libtorch_cuda_vendor** -- Optional CUDA LibTorch 2.9.1 overlay for C++
  conversions, supporting `cu126`, `cu128`, and `cu130`.
- **python3_torch_vendor** -- Official CPU Python Torch 2.9.1 provider.
- **python3_torch_cuda_vendor** -- Optional CUDA Python Torch 2.9.1 overlay.
  It selects an official `cu126`, `cu128`, or `cu130` wheel from the detected
  CUDA Toolkit; JetPack provides the required installation on Tegra.
- **tensor_msgs** -- DLPack-aligned `ExperimentalTensor.msg` definition.
- **onnxruntime_core_vendor** -- CPU-only ONNX Runtime 1.26.0 C++ provider.
- **onnxruntime_cuda_vendor** -- ONNX Runtime 1.26.0 C++ provider for CUDA 12.
- **python_onnxruntime_vendor** -- CPU-only ONNX Runtime 1.26.0 Python provider.
- **python_onnxruntime_cuda_vendor** -- ONNX Runtime 1.26.0 Python provider for
  CUDA 12.
- **onnxruntime_conversions** -- C++ `Ort::Value` API and plugin registry.
- **onnxruntime_conversions_cpu** -- C++ host-memory implementation.
- **onnxruntime_conversions_cuda** -- C++ CUDA implementation.
- **onnxruntime_conversions_py** -- Python `OrtValue` API and plugin registry.
- **onnxruntime_conversions_py_cpu** -- Python host-memory implementation.
- **onnxruntime_conversions_py_cuda** -- Python CUDA implementation.
- **torch_conversions** -- C++ `at::Tensor` API and runtime plugin registry.
- **torch_conversions_cpu** -- C++ host-memory implementation.
- **torch_conversions_cuda** -- C++ CUDA implementation backed by `cuda_buffer`.
- **torch_conversions_py** -- Python `torch.Tensor` API and plugin registry.
- **torch_conversions_py_cpu** -- Python host-memory implementation.
- **torch_conversions_py_cuda** -- Python CUDA implementation.

### Torch provider model

The C++ and Python conversion APIs are device-neutral within PyTorch, but they
are not framework-ABI-neutral: the C++ API exposes `at::Tensor`, and the Python
API imports `torch`. The conversion cores therefore depend on CPU Torch
providers, while the optional CUDA plugins bring CUDA-capable provider
overlays:

| Consumer | CPU provider | Optional CUDA provider |
| --- | --- | --- |
| `torch_conversions` | `libtorch_vendor` | `libtorch_cuda_vendor` |
| `torch_conversions_py` | `python3_torch_vendor` | `python3_torch_cuda_vendor` |

All four providers use exactly PyTorch 2.9.1. The CPU providers contain the
official CPU LibTorch archive and Python wheel and have no CUDA dependency.
The CUDA providers are separate packages that depend on their corresponding
CPU provider and on the supported CUDA dependency closure. Consequently,
installing either core plus its CPU plugin does not install CUDA.

Installing a CUDA conversion plugin later does not rebuild or replace the
conversion core or CPU provider. Its CUDA provider is installed under a
separate ROS-prefix directory, and ROS environment hooks place that directory
ahead of the CPU provider for subsequently started processes. The process then
loads one ABI-compatible set of Torch libraries, while both the CPU and CUDA
conversion plugins remain selectable. Restart the process and source the ROS
setup file after installing or removing a provider; already-loaded Torch
libraries and plugin registries cannot be switched safely in-process.

APT installs the providers transitively, so typical installations are:

```bash
# CPU-only C++ and Python conversions
sudo apt install \
  ros-$ROS_DISTRO-torch-conversions-cpu \
  ros-$ROS_DISTRO-torch-conversions-py-cpu

# Add CUDA later; this retains the CPU plugins
sudo apt install \
  ros-$ROS_DISTRO-torch-conversions-cuda \
  ros-$ROS_DISTRO-torch-conversions-py-cuda
```

### ONNX Runtime provider model

The C++ and Python conversion cores use the CUDA-capable ONNX Runtime 1.26.0
providers. The CPU-only C++ and Python providers remain available for
independent consumers and conflict with the corresponding CUDA-capable
provider packages.

| Consumer | Provider |
| --- | --- |
| `onnxruntime_conversions` | `onnxruntime_cuda_vendor` |
| `onnxruntime_conversions_py` | `python_onnxruntime_cuda_vendor` |

The conversion plugins are packaged separately from the cores. Host plugins
have priority 0 and accelerator plugins have priority 100. Applications can
select a backend per call or set `ROSIDL_TENSOR_BACKEND` before starting the
process.

## Deb build status

### ROS 2 Lyrical (Ubuntu Resolute)

| Package | Source deb | Binary deb (amd64) | Binary deb (arm64) |
| ------- | ---------- | ------------------ | ------------------ |
| cuda_buffer | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__cuda_buffer__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__cuda_buffer__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__cuda_buffer__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__cuda_buffer__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__cuda_buffer__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__cuda_buffer__ubuntu_resolute_arm64__binary/) |
| cuda_buffer_backend | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__cuda_buffer_backend__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__cuda_buffer_backend__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__cuda_buffer_backend__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__cuda_buffer_backend__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__cuda_buffer_backend__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__cuda_buffer_backend__ubuntu_resolute_arm64__binary/) |
| cuda_buffer_backend_msgs | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__cuda_buffer_backend_msgs__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__cuda_buffer_backend_msgs__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__cuda_buffer_backend_msgs__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__cuda_buffer_backend_msgs__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__cuda_buffer_backend_msgs__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__cuda_buffer_backend_msgs__ubuntu_resolute_arm64__binary/) |
| tensor_msgs | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__tensor_msgs__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__tensor_msgs__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__tensor_msgs__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__tensor_msgs__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__tensor_msgs__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__tensor_msgs__ubuntu_resolute_arm64__binary/) |
| torch_conversions | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lsrc_uR__torch_conversions__ubuntu_resolute__source)](https://build.ros2.org/job/Lsrc_uR__torch_conversions__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_uR64__torch_conversions__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Lbin_uR64__torch_conversions__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Lbin_armv8_uRv8__torch_conversions__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Lbin_armv8_uRv8__torch_conversions__ubuntu_resolute_arm64__binary/) |

### ROS 2 Rolling (Ubuntu Resolute)

| Package | Source deb | Binary deb (amd64) | Binary deb (arm64) |
| ------- | ---------- | ------------------ | ------------------ |
| cuda_buffer | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__cuda_buffer__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__cuda_buffer__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__cuda_buffer__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__cuda_buffer__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__cuda_buffer__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__cuda_buffer__ubuntu_resolute_arm64__binary/) |
| cuda_buffer_backend | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__cuda_buffer_backend__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__cuda_buffer_backend__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__cuda_buffer_backend__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__cuda_buffer_backend__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__cuda_buffer_backend__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__cuda_buffer_backend__ubuntu_resolute_arm64__binary/) |
| cuda_buffer_backend_msgs | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__cuda_buffer_backend_msgs__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__cuda_buffer_backend_msgs__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__cuda_buffer_backend_msgs__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__cuda_buffer_backend_msgs__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__cuda_buffer_backend_msgs__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__cuda_buffer_backend_msgs__ubuntu_resolute_arm64__binary/) |
| tensor_msgs | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__tensor_msgs__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__tensor_msgs__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__tensor_msgs__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__tensor_msgs__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__tensor_msgs__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__tensor_msgs__ubuntu_resolute_arm64__binary/) |
| torch_conversions | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rsrc_uR__torch_conversions__ubuntu_resolute__source)](https://build.ros2.org/job/Rsrc_uR__torch_conversions__ubuntu_resolute__source/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_uR64__torch_conversions__ubuntu_resolute_amd64__binary)](https://build.ros2.org/job/Rbin_uR64__torch_conversions__ubuntu_resolute_amd64__binary/) | [![Build Status](https://build.ros2.org/buildStatus/icon?job=Rbin_unv8_uRv8__torch_conversions__ubuntu_resolute_arm64__binary)](https://build.ros2.org/job/Rbin_unv8_uRv8__torch_conversions__ubuntu_resolute_arm64__binary/) |

## Prerequisites

- A ROS 2 Rolling development environment. See the upstream
  [Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
  guide for the canonical source-build flow, or use the pixi workflow
  shipped by the [`ros2/ros2`](https://github.com/ros2/ros2) meta-repo.
- CPU LibTorch and Python Torch 2.9.1 providers for CPU conversions.
- A CUDA Toolkit in the 12 or 13 series for the CUDA buffer and Torch
  accelerator packages, declared through the upstream `nvidia-cuda` rosdep
  key. The Torch vendors select a `cu126`, `cu128`, or `cu130` distribution. A
  Torch CPU-only installation does not require CUDA packages, a GPU, or a
  driver.
- ONNX Runtime accelerator packages require CUDA 12 and cuDNN 9.

Per-package build, test, and run details live in each package's README:

- [`cuda_buffer_backend/README.md`](cuda_buffer_backend/README.md)
- [`onnxruntime_conversions/README.md`](onnxruntime_conversions/README.md)
- [`torch_conversions/README.md`](torch_conversions/README.md)

## API overview

### CUDA buffer backend (`cuda_buffer_backend`)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

// Publisher: allocate + write directly to the output buffer.
sensor_msgs::msg::Image msg;
msg.data = cuda_buffer_backend::allocate_buffer(byte_count);
{
  auto wh = cuda_buffer_backend::from_output_buffer(msg.data, stream);
  uint8_t * out = wh.get_ptr();
  my_kernel<<<...>>>(out, ...);
}  // wh destructor records the write event on `stream`
publisher->publish(msg);

// Subscriber: input/read handle (waits on publisher's write event).
auto rh = cuda_buffer_backend::from_input_buffer(msg->data, stream);
use_data<<<...>>>(rh.get_ptr(), ...);  // rh.get_ptr() returns const uint8_t *

// Auto-promotion: passing a non-CUDA buffer allocates a fresh CUDA buffer
// and (for inputs) copies H2D;
auto rh = cuda_buffer_backend::from_input_buffer(cpu_or_other_buf, stream);
```

### Torch tensor API (`torch_conversions`)

```cpp
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

// Publisher: allocate a Tensor message (accelerated backend when available).
auto msg = torch_conversions::allocate_tensor_msg(
  /*shape=*/{1080, 1920, 3}, torch::kUInt8);

// Wrap as at::Tensor without copying and write into it. On an accelerator,
// pass the stream your kernels run on as a trailing argument so the conversion
// plugin orders its access against them.
at::Tensor t_out = torch_conversions::from_output_tensor_msg(*msg);
my_pipeline(t_out);
publisher->publish(std::move(msg));

// Subscriber: independent tensor by default.
at::Tensor t_in = torch_conversions::from_input_tensor_msg(*received_msg);
```

### ONNX Runtime tensor API (`onnxruntime_conversions`)

```cpp
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

auto msg = onnxruntime_conversions::allocate_tensor_msg(
  {1080, 1920, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "cuda");
{
  auto output = onnxruntime_conversions::from_output_tensor_msg(*msg, stream);
  my_pipeline(output.value());
}
publisher->publish(std::move(*msg));

auto input = onnxruntime_conversions::from_input_tensor_msg(
  received_msg, stream);
use_tensor(input.value());

onnxruntime_conversions::configure_session_options(
  options, "cuda", /*device_id=*/0, stream);
```

The message schema carries DLPack-aligned dtype, shape, stride, and offset
metadata, while device placement is derived from the underlying
`rosidl::Buffer` backend. Each framework conversion API loads its device
implementations from separately packaged plugins.

## License

Apache-2.0
