# rosidl_buffer_backends

CUDA buffer backend implementation for `rosidl::Buffer`, enabling zero-copy
GPU memory sharing between ROS 2 publishers and subscribers, plus tensor
conversion libraries that build on the same buffer infrastructure.

## Packages

- **cuda_buffer** -- Core CUDA buffer library (VMM-backed IPC memory pool,
  host endpoint manager, ReadHandle/WriteHandle with CUDA event sync).
- **cuda_buffer_py** -- Python CUDA buffer allocation and scoped read/write
  handles for rclpy publishers and subscribers.
- **cuda_buffer_backend** -- BufferBackend plugin for CUDA IPC transport.
- **cuda_buffer_backend_msgs** -- ROS 2 message definitions for CUDA buffer
  descriptors.
- **libtorch_vendor** -- Vendor package that downloads and installs the
  pre-built LibTorch C++ distribution.
- **pytorch_vendor** -- Vendor package that installs the Python Torch
  distribution using the same platform and CUDA selection policy as
  `libtorch_vendor`.
- **tensor_msgs** -- DLPack-aligned `ExperimentalTensor.msg` definition.
- **onnxruntime_core_vendor** -- CUDA-neutral ONNX Runtime headers, core
  runtime, and shared provider support extracted from an official GPU archive.
- **onnxruntime_cuda_vendor** -- Optional CUDA execution-provider
  library installed beside the canonical core runtime.
- **python_onnxruntime_vendor** -- Unmodified CPU-only Python ONNX Runtime
  wheel packaged for ROS.
- **python_onnxruntime_cuda_vendor** -- Unmodified CUDA Python ONNX Runtime
  wheel packaged for ROS.
- **onnxruntime_conversions** -- Compiled C++ conversion library and plugin
  registry, including its required runtime-discovered CPU plugin.
- **onnxruntime_conversions_cuda_plugin** -- Optional runtime-discovered CUDA
  storage and execution-provider backend.
- **onnxruntime_conversions_py_core** -- Vendor-neutral Python ROS package under the shared
  `onnxruntime_conversions/` source container, providing CPU and CUDA
  conversions using NumPy views or ONNX Runtime's public DLPack protocol.
- **onnxruntime_conversions_py_cpu** -- User-facing CPU Python conversion
  runtime metapackage and apt package
  `ros-$ROS_DISTRO-onnxruntime-conversions-py-cpu`.
- **onnxruntime_conversions_py_cuda** -- CUDA Python conversion runtime
  metapackage.
- **torch_conversions** -- Header-only helper library that converts between
  `tensor_msgs/ExperimentalTensor` and `at::Tensor` and exposes DLPack import /
  export. Replaces the older `torch_buffer_backend` plugin approach with a
  plain message + bridge library that rides on top of whichever
  `rosidl::Buffer` backend is registered (CUDA when available, CPU
  otherwise).
- **torch_conversions_py** -- Python CPU conversions and optional, lazily
  loaded CUDA buffer support for `tensor_msgs/ExperimentalTensor`.

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
- CUDA Toolkit on the host for CUDA packages.

## ONNX Runtime architecture

`onnxruntime_core_vendor` always installs a CUDA-neutral SDK and runtime. Its
CUDA 12 or CUDA 13 setting selects only the x86_64 GPU archive used as the
source of those core files; it does not add a CUDA dependency or enable CUDA
for consumers. On arm64, including JetPack hosts, it uses the official
ONNX Runtime CPU archive. `onnxruntime_cuda_vendor` is the optional owner of
`libonnxruntime_providers_cuda.so`. ONNX Runtime discovers that provider plugin
beside the physical `libonnxruntime.so`, so the two vendors must be installed
into one merged prefix.

ARM64 supports the C++ CPU conversion path. The Python CPU path additionally
requires Python 3.11 or newer for ONNX Runtime 1.28, so it does not support
JetPack 6's default Python 3.10. JetPack GPU execution is not currently
provided because Microsoft does not publish a matching arm64 GPU C++ archive;
JetPack itself supplies CUDA and cuDNN but not the ONNX Runtime C++ SDK.

`python_onnxruntime_vendor` installs only the CPU `onnxruntime` wheel.
`python_onnxruntime_cuda_vendor` installs the matching `onnxruntime-gpu` wheel;
`PYTHON_ONNXRUNTIME_CUDA_VENDOR_VARIANT` selects its CUDA 12 or CUDA 13
distribution. Both wheels are installed unchanged and retain their bundled
native libraries. They are independent of the C++ ONNX Runtime vendors.
The two Python vendors declare a package conflict because they own the same
`onnxruntime` import path. The CUDA wheel already contains the CPU execution
provider.

The vendored Python runtime is constrained to the Python ABI, operating system,
and architecture of its wheel. No external ONNX Runtime installation is
required.

Install `onnxruntime_conversions_py_cpu` for CPU Python use or
`onnxruntime_conversions_py_cuda` for CUDA Python use; no package depends on
both vendors. C++ conversion backends are discovered at runtime and remain
separate from the Python wheel runtime.

```bash
# CPU Python
sudo apt update
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cpu

# CUDA Python (install instead of the CPU metapackage)
sudo apt update
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cuda

# Source builds; use separate install prefixes for CPU and CUDA Python vendors
colcon build --merge-install --packages-up-to onnxruntime_conversions_py_cpu
colcon build --merge-install --packages-up-to onnxruntime_conversions_py_cuda
```

Python conversion packages register storage adapters through the ament index.
The CPU adapter is always available, while
`onnxruntime_conversions_py_cuda` registers a higher-priority CUDA adapter and
makes CUDA the allocation default. Additional platform packages can register
other device and buffer backend IDs without changing the core package. CUDA
allocations require a non-null explicit stream. Explicit device overrides
remain strict, and existing messages dispatch from their buffer backend.
Adapter packages install a resource named after the package under
`onnxruntime_conversions__adapters`; its contents are a Python
`module:register_function` entry point.

For C++, requesting a missing backend throws
`ONNX Runtime conversion backend '<name>' is unavailable.` followed by loaded
backend IDs, discoverable plugin classes, and exact plugin load failures. A
missing required CPU plugin reports
`Required onnxruntime_conversions CPU plugin is unavailable.` with the same
details.

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
auto guard = torch_conversions::set_stream();
auto msg = torch_conversions::allocate_tensor_msg(
  /*shape=*/{1080, 1920, 3}, torch::kUInt8);

// Wrap as at::Tensor without copying and write into it.
at::Tensor t_out = torch_conversions::from_output_tensor_msg(*msg);
my_pipeline(t_out);
publisher->publish(std::move(msg));

// Subscriber: independent tensor by default.
auto guard = torch_conversions::set_stream();
at::Tensor t_in = torch_conversions::from_input_tensor_msg(*received_msg);
```

The message schema carries DLPack-compatible dtype, shape, stride, and offset
metadata, while device placement is derived from the underlying
`rosidl::Buffer` backend. ONNX Runtime's public C++ API has no direct
`from_dlpack` operation. The ONNX Runtime conversion packages validate the
message metadata and wrap its CPU or CUDA pointer without copying by calling
`Ort::Value::CreateTensor`.

## License

Apache-2.0
