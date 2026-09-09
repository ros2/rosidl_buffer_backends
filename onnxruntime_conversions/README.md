# ONNX Runtime conversions

Zero-copy conversions between `tensor_msgs/msg/ExperimentalTensor` and ONNX
Runtime 1.23.2. CUDA uses CUDA 12 and cuDNN 9.

`onnxruntime_conversions` translates between ONNX Runtime tensors and DLPack,
and [`dlpack_conversions`](../dlpack_conversions/README.md) owns the message
storage behind it. Which memory a tensor lands in is decided by the storage
plugin installed alongside the adapter, so the same code runs on the host or
on a GPU depending on what is installed. Pass a backend name to choose among
several, or set `ROSIDL_TENSOR_BACKEND` to pick one for the whole process.

| Device | C++ plugin | Python plugin |
| --- | --- | --- |
| Host | `dlpack_conversions_cpu` | `dlpack_conversions_py_cpu` |
| CUDA | `dlpack_conversions_cuda` | `dlpack_conversions_py_cuda` |

The conversions never ask for an `Ort::MemoryInfo`, because the plugin that
allocated the storage already knows where it lives.

## C++

The adapter is header-only and compiles against whichever ONNX Runtime the
consumer already resolved, so it does not pin a runtime version.

Debian:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions \
  ros-$ROS_DISTRO-dlpack-conversions-cuda
```

Source:

```bash
colcon build --merge-install --packages-up-to onnxruntime_conversions \
  dlpack_conversions_cuda
```

### Publisher

```cpp
#include <cuda_runtime.h>
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

auto msg = onnxruntime_conversions::allocate_tensor_msg(
  {480, 640, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
{
  // The view holds the storage lease, so keep it alive while ONNX Runtime
  // reads from or writes to the value.
  auto output = onnxruntime_conversions::from_output_tensor_msg(*msg, stream);
  my_pipeline(output.value());
}
publisher->publish(std::move(*msg));
```

### Subscriber

```cpp
void callback(const onnxruntime_conversions::TensorMsg & received)
{
  auto input = onnxruntime_conversions::from_input_tensor_msg(
    received, stream);
  my_pipeline(input.value());
}
```

### Existing tensor

```cpp
auto msg = onnxruntime_conversions::to_tensor_msg(ort_value, stream);
publisher->publish(std::move(*msg));
```

`to_tensor_msg(*msg, ort_value, stream)` reuses a pre-sized message. The copy
is performed by the plugin that owns the memory, so a device value never
stages through the host.

### CUDA session

```cpp
Ort::SessionOptions options;
onnxruntime_conversions::configure_session_options(
  options, "cuda", /*device_id=*/0, stream);
Ort::Session session(env, model, model_size, options);
```

This appends the execution provider that runs where the named backend
allocates, so the session reads message storage in place. Backends ONNX
Runtime ships no provider for are rejected rather than silently run on the
host; append your own provider in that case.

## Python

Debian:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py \
  ros-$ROS_DISTRO-dlpack-conversions-py-cuda
```

Source:

```bash
colcon build --merge-install --packages-up-to onnxruntime_conversions_py \
  dlpack_conversions_py_cuda
```

### Publisher

```python
import numpy as np
from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg

msg = allocate_tensor_msg((480, 640, 3), np.float32, 'cuda')
with from_output_tensor_msg(msg, stream) as output:
    my_pipeline(output)
publisher.publish(msg)
```

### Subscriber

```python
from onnxruntime_conversions import from_input_tensor_msg

def callback(received):
    with from_input_tensor_msg(received, stream) as value:
        my_pipeline(value)
```

Both conversions return `None` when the message carries no storage.

### Existing tensor

```python
from onnxruntime_conversions import to_tensor_msg

msg = to_tensor_msg(ort_value, stream=stream)
publisher.publish(msg)
```

`to_tensor_msg(msg, ort_value, stream)` reuses a pre-sized message.

### Session providers

```python
from onnxruntime_conversions import session_providers

session = ort.InferenceSession(
    model, providers=session_providers('cuda', 0, stream))
```

## Version

Require ONNX Runtime 1.23.x. Remove other copies from `/usr/local`, pip,
Conda, or `PYTHONPATH` if they take precedence.
