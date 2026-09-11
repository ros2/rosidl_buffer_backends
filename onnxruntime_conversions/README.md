# ONNX Runtime conversions

Conversions between `tensor_msgs/msg/ExperimentalTensor` and ONNX Runtime
tensors. The public packages are framework-specific and device-neutral:

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `onnxruntime_conversions` | `onnxruntime_conversions_py` |
| CPU implementation | `onnxruntime_conversions_cpu` | `onnxruntime_conversions_py_cpu` |
| CUDA implementation | `onnxruntime_conversions_cuda` | `onnxruntime_conversions_py_cuda` |

Install a CPU or CUDA plugin Debian to choose the available implementations
without changing application code or rebuilding the core. C++ discovers plugins
on first registry use; Python discovers them when the module is imported.
Restart the application after installing a plugin or provider and source the
ROS environment before starting it.

Default allocation and session setup use the available plugin with the highest
priority: CPU is 0 and CUDA is 100. Set `ROSIDL_TENSOR_BACKEND=cpu` or `cuda` to
override that default, or pass a backend explicitly. Views use the message's
storage backend; copies into a new message use the source tensor's device
unless the Python caller specifies a destination backend.

## ONNX Runtime provider boundary

All four providers use ONNX Runtime 1.26.0. The accelerator providers require
CUDA 12 and cuDNN 9.

The C++ core and both plugins depend on `onnxruntime_cuda_vendor`; the Python
core depends on `python_onnxruntime_cuda_vendor`. Installing only the CPU
conversion plugin therefore still includes CUDA user-space dependencies,
although CPU execution needs no GPU or host driver. Adding the CUDA plugin
enables device conversions using the same framework provider.

The CPU-only providers, `onnxruntime_core_vendor` and
`python_onnxruntime_vendor`, serve independent consumers. Each conflicts with
its corresponding CUDA provider and is not used by these conversion cores.

## Storage lifetime

C++ CPU views borrow message storage: keep the message alive and do not resize
its buffer while a view exists. CUDA views retain a buffer access handle.
Release output views before publishing so their write handles can record
completion on the execution stream. Python views retain their backing storage;
use a context manager to release the view after use.

## C++

Install CPU support:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions \
  ros-$ROS_DISTRO-onnxruntime-conversions-cpu
```

Add CUDA support later:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-cuda
```

Build from source:

```bash
rosdep install --from-paths . --ignore-src -y
colcon build --merge-install --packages-up-to \
  onnxruntime_conversions_cpu onnxruntime_conversions_cuda
```

Create and wrap message storage without copying:

```cpp
#include <cuda_runtime.h>
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

auto msg = onnxruntime_conversions::allocate_tensor_msg(
  {480, 640, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
{
  auto output = onnxruntime_conversions::from_output_tensor_msg(*msg, stream);
  my_pipeline(output.value());
}  // releases the storage lease
publisher->publish(std::move(*msg));
```

For received messages and existing values:

```cpp
auto input = onnxruntime_conversions::from_input_tensor_msg(received, stream);
my_pipeline(input.value());

auto outgoing = onnxruntime_conversions::to_tensor_msg(ort_value, stream);
onnxruntime_conversions::to_tensor_msg(*preallocated, ort_value, stream);
```

Bind ONNX Runtime's CUDA execution provider to the same application stream:

```cpp
Ort::SessionOptions options;
onnxruntime_conversions::configure_session_options(
  options, "cuda", /*device_id=*/0, stream);
Ort::Session session(env, model, model_size, options);
```

The C++ plugin constructs an `Ort::Value` over the storage pointer with
`Ort::Value::CreateTensor`.

## Python

Install CPU support, then optionally add CUDA:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py \
  ros-$ROS_DISTRO-onnxruntime-conversions-py-cpu
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cuda
```

Build from source:

```bash
rosdep install --from-paths . --ignore-src -y
colcon build --merge-install --packages-up-to \
  onnxruntime_conversions_py_cpu onnxruntime_conversions_py_cuda
```

```python
import onnxruntime as ort
from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg
from onnxruntime_conversions import session_providers
from onnxruntime_conversions import to_tensor_msg

msg = allocate_tensor_msg((480, 640, 3), 1, 'cuda')
with from_output_tensor_msg(msg, stream) as output:
    my_pipeline(output)

with from_input_tensor_msg(msg, stream) as value:
    consume(value)

outgoing = to_tensor_msg(ort_value, stream=stream)
session = ort.InferenceSession(
    model, providers=session_providers('cuda', 0, stream))
```

Python plugins use `OrtValue.from_dlpack` with a private capsule helper that
retains the storage and its access handle.

## License

Apache-2.0
