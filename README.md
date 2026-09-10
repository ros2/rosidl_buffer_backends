# rosidl_buffer_backends

CUDA buffer backend implementation for `rosidl::Buffer`, enabling zero-copy
GPU memory sharing between ROS 2 publishers and subscribers, plus tensor
conversion libraries that build on the same buffer infrastructure.

## Packages

- **cuda_buffer** -- Core CUDA buffer library (VMM-backed IPC memory pool,
  host endpoint manager, ReadHandle/WriteHandle with CUDA event sync).
- **cuda_buffer_py** -- Python CUDA buffer allocation and scoped read/write
  handles for rclpy publishers and subscribers.
- **[cuda_buffer_backend](cuda_buffer_backend/README.md)** -- BufferBackend
  plugin for CUDA IPC transport.
- **cuda_buffer_backend_msgs** -- ROS 2 message definitions for CUDA buffer
  descriptors.
- **libtorch_vendor** -- CUDA LibTorch 2.9.1 vendor for C++ conversions,
  supporting `cu126`, `cu128`, and `cu130`.
- **python3_torch_cuda_vendor** -- CUDA Python Torch vendor. It reuses a compatible
  PyTorch 2.9.1 installation or selects an official `cu126`, `cu128`, or
  `cu130` wheel from the detected CUDA Toolkit; JetPack provides the
  required installation on Tegra.
- **tensor_msgs** -- DLPack-aligned `ExperimentalTensor.msg` definition.
- **onnxruntime_core_vendor** -- ONNX Runtime 1.29.0 CPU and CUDA-neutral C++
  distribution.
- **onnxruntime_cuda_vendor** -- ONNX Runtime GPU 1.29.0 C++ distribution
  matching the host CUDA major.
- **python_onnxruntime_vendor** -- ONNX Runtime 1.29.0 CPU Python wheel
  packaged for ROS.
- **python_onnxruntime_cuda_vendor** -- Unmodified CUDA Python ONNX Runtime
  wheel packaged for ROS.
- **[dlpack_conversions](dlpack_conversions/README.md)** -- Framework-free C++
  core. Allocates message storage, hands out DLPack tensors over it, and loads
  storage plugins.
- **dlpack_conversions_cpu** -- Host memory storage plugin.
- **dlpack_conversions_cuda** -- CUDA device memory storage plugin, backed by
  `cuda_buffer`.
- **dlpack_conversions_py** -- Framework-free Python core and plugin registry.
- **dlpack_conversions_py_cpu** -- Host memory storage plugin for Python.
- **dlpack_conversions_py_cuda** -- CUDA storage plugin for Python.
- **[onnxruntime_conversions](onnxruntime_conversions/README.md)** --
  Header-only C++ adapter between ONNX Runtime `Ort::Value` tensors and the
  DLPack core.
- **onnxruntime_conversions_py** -- Python adapter between ONNX Runtime
  `OrtValue` tensors and the DLPack core.
- **[torch_conversions](torch_conversions/README.md)** -- Header-only C++
  adapter between PyTorch tensors and the DLPack core.
- **torch_conversions_py** -- Python adapter between PyTorch tensors and the
  DLPack core.

Storage lives behind a plugin interface that speaks only DLPack, so the
adapters carry no device code and no framework pins a device. Install an
adapter with whichever storage plugins the machine should support; adding
`dlpack_conversions_cuda` later moves existing code onto the GPU without
rebuilding it. Name a backend per call, or set `ROSIDL_TENSOR_BACKEND` to
choose one for the whole process.

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
- A CUDA Toolkit in the 12 or 13 series for the CUDA buffer, Torch, and ONNX
  Runtime packages, declared through the `cuda-toolkit` rosdep key. The
  vendors read only the major version, choosing `cu126`, `cu128`, or `cu130`
  wheels and CUDA 12 or CUDA 13 ONNX Runtime archives. CPU-only conversions
  do not require CUDA.

Torch conversions use Ubuntu Resolute's `libtorch-dev` and `python3-torch` on
CPU, and `libtorch_vendor` and `python3_torch_cuda_vendor` on CUDA; each
vendor conflicts with the distro package it replaces. ONNX Runtime is
vendored at 1.29.0 on both paths, because Ubuntu ships 1.23.2, which predates
the `OrtValue.from_dlpack` the Python adapter needs.

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
// pass the stream your kernels run on as a trailing argument so the storage
// plugin orders its access against them, the same way
// onnxruntime_conversions takes an execution stream.
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

// Publisher: allocate a Tensor message and write through an Ort::Value view.
// The storage plugin decides where the memory lives, so no Ort::MemoryInfo
// is needed here.
auto msg = onnxruntime_conversions::allocate_tensor_msg(
  {1080, 1920, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "cuda");
{
  auto t_out = onnxruntime_conversions::from_output_tensor_msg(*msg, stream);
  my_pipeline(t_out.value());
}  // t_out releases the storage lease
publisher->publish(std::move(*msg));

// Subscriber: wrap the received message as Ort::Value.
auto t_in = onnxruntime_conversions::from_input_tensor_msg(
  received_msg, stream);
use_tensor(t_in.value());

// CUDA session: bind the application stream before creating the session.
onnxruntime_conversions::configure_session_options(
  options, "cuda", /*device_id=*/0, stream);
```

The message schema carries DLPack-compatible dtype, shape, stride, and offset
metadata, while device placement follows the storage plugin that allocated the
message. ONNX Runtime has no public `from_dlpack` C++ API; the conversion
wraps the message pointer with `Ort::Value::CreateTensor`.

## License

Apache-2.0
