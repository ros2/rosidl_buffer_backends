# PyTorch conversions

Conversions between `tensor_msgs/msg/ExperimentalTensor` and PyTorch tensors.
The public packages are framework-specific and device-neutral:

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `torch_conversions` | `torch_conversions_py` |
| CPU implementation | `torch_conversions_cpu` | `torch_conversions_py_cpu` |
| CUDA implementation | `torch_conversions_cuda` | `torch_conversions_py_cuda` |

Install CPU and accelerator plugins as separate Debian packages without
rebuilding the conversion core or a device-neutral application. C++ discovers
plugins on first registry use; Python discovers them when the module is imported.
Restart the application after installing a plugin or provider and source the
ROS environment before starting it.

Default allocation uses the available plugin with the highest priority: CPU
is 0 and CUDA is 100. Set `ROSIDL_TENSOR_BACKEND=cpu` or `cuda` to override that
default, or pass a device to `allocate_tensor_msg`. Views use the message's
storage backend; copies into a new message use the source tensor's device.
An explicit device such as `cuda:1` preserves its device index. Additional
accelerators can provide plugins without adding device-specific logic to the
core; their framework providers must satisfy the core's ABI contract.

## Dependencies and source builds

Debians target Ubuntu 26.04 (Resolute) and bundle Torch **2.14.0**; sourcing ROS
selects that installation. Build from source to reuse an existing Torch
installation or on earlier Ubuntu releases. Ubuntu 24.04, including JetPack,
also requires [ROS Lyrical built from source](https://github.com/ros2/ros2_documentation/blob/lyrical/source/Releases/lyrical/supported-platforms.rst).

Source builds reuse detected CUDA and **Torch/LibTorch >=2.5.0**. CUDA plugins
require CUDA-enabled Torch. If no compatible installation is found, the fallback
is **2.14.0**; its CUDA build uses `cu130` and requires an installed
**CUDA >=13.1,<14** toolkit.

From this repository's root, after sourcing ROS, build with an existing CUDA
toolkit (including on Ubuntu before Resolute):

```bash
rosdep install --from-paths torch_vendor torch_conversions \
  tensor_msgs cuda_buffer_backend --ignore-src -y --skip-keys cuda-toolkit
colcon build --merge-install --packages-up-to \
  torch_conversions_cpu torch_conversions_cuda \
  torch_conversions_py_cpu torch_conversions_py_cuda
source install/setup.bash
```

On Resolute, omit `--skip-keys cuda-toolkit` to install the toolkit through rosdep/APT.
Append `--cmake-args` to `colcon build` with any of these overrides:

- `-DFORCE_BUILD_VENDOR_PKG=ON`: select the pinned fallback.
- `-DTorch_DIR=/path/to/libtorch/share/cmake/Torch`: select a C++ SDK.
- `-DPython3_EXECUTABLE=/path/to/python`: select Python; its major/minor version must match ROS.
- `-DCUDAToolkit_ROOT=/path/to/cuda`: select the CUDA toolkit.

C++ requires C++20 and the libstdc++ C++11 ABI. Use fresh build directories and
rebuild native vendors, conversions and applications together when changing the
Torch release.

## Execution streams

Conversions use Torch's current stream on the buffer or source tensor's device
unless a stream is passed explicitly. Applications already managing a Torch
stream do not need `set_stream()`.

Use `auto guard = torch_conversions::set_stream()` in C++, or `with set_stream():`
in Python, to select a stream. CPU is a no-op; CUDA selects a pooled stream.
The previous stream and device are restored when the scope exits. Keep
allocation, conversions, tensor operations, and publication inside the scope.

The guard does not synchronize on exit; synchronize explicitly when using
different streams. Keep caller-owned streams alive until all views are released
and queued work finishes. Copies complete before returning; zero-copy views and
input clones may still have queued GPU work.

## C++

CPU installation:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions \
  ros-$ROS_DISTRO-torch-conversions-cpu
```

Add CUDA later:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-cuda
```

For a CPU-only source build, restrict rosdep to the CPU source packages as
well as omitting the CUDA target:

```bash
rosdep install --from-paths \
  tensor_msgs \
  torch_vendor/libtorch_vendor \
  torch_conversions/torch_conversions \
  torch_conversions/torch_conversions_cpu \
  --ignore-src -y
colcon build --merge-install --packages-up-to torch_conversions_cpu
```

### Publisher

These examples assume valid, nonempty messages. Select a stream for the
publisher's conversion and pipeline operations:

```cpp
#include "torch_conversions/torch_conversions.hpp"

auto guard = torch_conversions::set_stream();
auto msg = torch_conversions::allocate_tensor_msg(
  {480, 640, 3}, torch::kUInt8);
auto output = torch_conversions::from_output_tensor_msg(*msg);
produce(output);
publisher->publish(*msg);
```

The guard covers allocation, conversion, and the application's `produce()`,
which fills the output in place. Do not use `output` for further writes after
publishing.

### Subscriber

Select a stream inside the subscriber callback:

```cpp
#include "torch_conversions/torch_conversions.hpp"

auto guard = torch_conversions::set_stream();
auto input = torch_conversions::from_input_tensor_msg(*received, /*clone=*/false);
consume(input);
```

The guard covers both the input conversion and the application's read-only
`consume()`. Omit `/*clone=*/false` to get an independent tensor instead of a
zero-copy view.

### Existing tensors

Use `to_tensor_msg()` only when copying an already-existing tensor into a
message. Run these calls on the tensor producer's current stream:

```cpp
auto outgoing = torch_conversions::to_tensor_msg(tensor);
publisher->publish(std::move(outgoing));
```

Copies complete before returning. If the tensor was produced on another
stream, synchronize with it or perform the copy on that stream.

## Python

CPU installation:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-py \
  ros-$ROS_DISTRO-torch-conversions-py-cpu
```

Add CUDA later:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-py-cuda
```

For a Python CPU-only source build, restrict both commands to the CPU stack:

```bash
rosdep install --from-paths \
  tensor_msgs \
  torch_vendor/python3_torch_vendor \
  torch_conversions/torch_conversions_py \
  torch_conversions/torch_conversions_py_cpu \
  --ignore-src -y
colcon build --merge-install --packages-up-to torch_conversions_py_cpu
```

### Publisher

Select a stream for the publisher's conversion and pipeline operations:

```python
import torch
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import set_stream

with set_stream():
    msg = allocate_tensor_msg((480, 640, 3), torch.uint8)
    output = from_output_tensor_msg(msg)
    produce(output)
    publisher.publish(msg)
```

The stream scope covers allocation, conversion, and the application's
`produce()`, which fills the output in place. Do not use `output` for further
writes after publishing.

### Subscriber

Select a stream inside the subscriber callback:

```python
from torch_conversions import from_input_tensor_msg
from torch_conversions import set_stream

with set_stream():
    input_tensor = from_input_tensor_msg(received, clone=False)
    consume(input_tensor)
```

The stream scope covers both the input conversion and the application's
read-only `consume()`. Omit `clone=False` to get an independent tensor instead
of a zero-copy view.

`with set_stream():` selects a stream; it is not a view context manager.
Conversions use that current stream without needing a `stream=` argument.
If the application already manages the current Torch stream, use its existing
scope instead.

### Existing tensors

Use `to_tensor_msg()` only when copying an already-existing tensor into a
message. Run these calls on the tensor producer's current stream:

```python
from torch_conversions import to_tensor_msg

outgoing = to_tensor_msg(tensor)
publisher.publish(outgoing)
```

Copies complete before returning. If the tensor was produced on another
stream, synchronize with it or perform the copy on that stream.

## License

Apache-2.0
