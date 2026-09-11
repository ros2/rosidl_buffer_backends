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
share LibTorch's C++ ABI. They all depend on the single `libtorch_vendor`
provider built from the CUDA-enabled LibTorch 2.9.1 distribution. Likewise,
the Python core depends on the single `python3_torch_cuda_vendor` provider.
The CPU and CUDA plugins never load different Torch distributions into one
process.

This means a CPU-only installation still installs the CUDA-capable provider
and its user-space toolkit libraries, although it does not require a GPU or
host driver. Adding the CUDA plugins later changes only which implementations
are discoverable. It does not exchange the provider.

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

Source build:

```bash
colcon build --merge-install --packages-up-to \
  torch_conversions_cpu torch_conversions_cuda
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

Source build:

```bash
colcon build --merge-install --packages-up-to \
  torch_conversions_py_cpu torch_conversions_py_cuda
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

## Validation

The release-like validation under `docker/resolute/debian/` builds every
Debian without a GPU, driver, preinstalled CUDA, LibTorch, or Python Torch.
A pristine consumer then installs only the cores and CPU plugins, builds the
test sources separately, manually runs the CPU unit and launch tests, installs
the CUDA plugins, verifies that the core files did not change, and reruns the
C++ and Python unit and launch tests on a GPU.

## License

Apache-2.0
