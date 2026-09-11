# PyTorch conversions

Conversions between `tensor_msgs/ExperimentalTensor` and PyTorch tensors.
The public packages are framework-specific and device-neutral:

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `torch_conversions` | `torch_conversions_py` |
| CPU implementation | `torch_conversions_cpu` | `torch_conversions_py_cpu` |
| CUDA implementation | `torch_conversions_cuda` | `torch_conversions_py_cuda` |

The core discovers implementations through the ament index. Installing the
CUDA plugin later does not replace or rebuild the core Debian. A newly started
process sees both plugins and can select CPU or CUDA per call; set
`ROSIDL_TENSOR_BACKEND=cpu` or `cuda` to choose the default. Discovery occurs
once when a process first uses the registry, so a process already running while
a Debian is installed must be restarted.

## Torch provider boundary

The C++ API exposes `at::Tensor`, so the core and both plugins necessarily
share LibTorch's C++ ABI. A CPU installation uses `libtorch_vendor`, which
installs the official CPU LibTorch 2.9.1 distribution. The CUDA plugin depends
on `libtorch_cuda_vendor`, which installs a matching CUDA LibTorch 2.9.1 tree
and places it ahead of the system libraries for newly started ROS processes.
Only one set of LibTorch SONAMEs is loaded in a process.

Python follows the same layout: `python3_torch_vendor` installs the official
CPU Torch wheel, while the CUDA plugin installs `python3_torch_cuda_vendor` as a
higher-priority ROS-prefix overlay. After the CUDA overlay is installed, both
CPU and CUDA conversion plugins use that CUDA-capable Torch distribution.

Consequently, CPU-only installations pull no CUDA dependencies. Adding the
CUDA plugins later changes the active provider and plugin discovery for new
processes without rebuilding or replacing either conversion core.

The dependency boundary is:

| Package | Required framework package | CUDA dependency |
| --- | --- | --- |
| `torch_conversions` | `libtorch_vendor` | No |
| `torch_conversions_cpu` | inherited from `torch_conversions` | No |
| `torch_conversions_cuda` | `libtorch_cuda_vendor` | Yes |
| `torch_conversions_py` | `python3_torch_vendor` | No |
| `torch_conversions_py_cpu` | inherited from `torch_conversions_py` | No |
| `torch_conversions_py_cuda` | `python3_torch_cuda_vendor` | Yes |

The CUDA providers depend on their CPU counterparts so both package sets can
coexist, but they do not link the two Torch distributions into one process.
ROS environment hooks prepend the CUDA provider directories after the CUDA
packages are installed. The exact shared-library ABI and Python package
version are pinned to 2.9.1 on both sides of the overlay.

Provider selection happens before Torch or the conversion registry is first
loaded. After changing the installed provider set, start a new process and
source `/opt/ros/$ROS_DISTRO/setup.bash`. Installing a Debian cannot replace
Torch safely inside an already-running process.

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

The explicit core package in the first command is optional because APT also
installs it transitively from `torch_conversions_cpu`.

Source build:

```bash
rosdep install --from-paths . --ignore-src -y
colcon build --merge-install --packages-up-to \
  torch_conversions_cpu torch_conversions_cuda
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

```cpp
#include <c10/cuda/CUDAStream.h>
#include "torch_conversions/torch_conversions.hpp"

void * stream = c10::cuda::getCurrentCUDAStream().stream();
auto msg = torch_conversions::allocate_tensor_msg(
  {480, 640, 3}, torch::kUInt8, c10::kCUDA);
at::Tensor output = torch_conversions::from_output_tensor_msg(*msg, stream);
my_pipeline(output);
publisher->publish(std::move(msg));
```

### Subscriber and existing tensors

```cpp
auto tensor = torch_conversions::from_input_tensor_msg(
  *received, /*clone=*/true, stream);
auto view = torch_conversions::from_input_tensor_msg(
  *received, /*clone=*/false, stream);
auto outgoing = torch_conversions::to_tensor_msg(tensor, stream);
torch_conversions::to_tensor_msg(*preallocated, tensor, stream);
```

Every conversion accepts an optional execution stream. Pass the stream when
work does not run on the default CUDA stream so buffer access is ordered
against the caller's kernels. Host calls can omit it.

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

As with C++, the explicit Python core package is optional when installing its
CPU plugin.

Source build:

```bash
rosdep install --from-paths . --ignore-src -y
colcon build --merge-install --packages-up-to \
  torch_conversions_py_cpu torch_conversions_py_cuda
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

```python
import torch
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_input_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import set_stream
from torch_conversions import to_tensor_msg

with set_stream('cuda'):
    msg = allocate_tensor_msg((480, 640, 3), torch.uint8, 'cuda')
    output = from_output_tensor_msg(msg)
    my_pipeline(output)

view = from_input_tensor_msg(msg, clone=False)
outgoing = to_tensor_msg(torch.arange(12, device='cuda').reshape(3, 4))
```

Python uses Torch's current CUDA stream when no explicit `stream=` integer is
provided. `to_tensor_msg(msg, tensor)` reuses preallocated message storage. The
CUDA plugin uses a private capsule bridge only to construct Torch zero-copy
views; it is not a public or framework-neutral conversion API.

## License

Apache-2.0
