# ONNX Runtime conversions

Conversions between `tensor_msgs/msg/ExperimentalTensor` and ONNX Runtime
tensors. The public packages are framework-specific and device-neutral:

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `onnxruntime_conversions` | `onnxruntime_conversions_py` |
| CPU implementation | `onnxruntime_conversions_cpu` | `onnxruntime_conversions_py_cpu` |
| CUDA implementation | `onnxruntime_conversions_cuda` | `onnxruntime_conversions_py_cuda` |

The core discovers implementations through pluginlib in C++ and an ament
resource index in Python. Installing the CUDA plugin later does not replace or
rebuild the core Debian. A newly started process sees both plugins and can
select CPU or CUDA per call; set `ROSIDL_TENSOR_BACKEND=cpu` or `cuda` to
choose the default. Discovery happens the first time a process uses the
registry, so restart a process that was already running while a plugin Debian
was installed.

## ONNX Runtime provider boundary

The C++ API exposes `Ort::Value`, so the core and both plugins must use one
process-wide ONNX Runtime ABI. They all depend on the CUDA-capable
`onnxruntime_cuda_vendor`. The Python core similarly depends on the single
`python_onnxruntime_cuda_vendor`. CPU and CUDA plugins never load competing
ONNX Runtime distributions into one process.

A CPU-only installation therefore includes the CUDA-capable provider and its
user-space CUDA libraries, but it needs no GPU or host driver. The shared CUDA
provider declares Resolute's `nvidia-cudnn` rosdep package for both C++ and
Python consumers. Adding conversion plugins later changes device discovery,
not the framework provider.

All four provider packages are pinned to ONNX Runtime 1.26.0 and support CUDA
12. Version 1.26 is the first release with the DLPack methods used internally
by the Python conversion plugins and the last stable PyPI release for CUDA 12.

### Why the Ubuntu package is not used

Ubuntu Resolute provides ONNX Runtime 1.23.2. Its Python `OrtValue` lacks
`from_dlpack`, `__dlpack__`, and `__dlpack_device__`, and its runtime has no
CUDA execution provider. The Python zero-copy conversion and CUDA plugins
therefore require the pinned upstream 1.26.0 distributions. The Ubuntu C++
development package is suitable for CPU-only consumers, but using it in the
conversion core would give the CPU and CUDA plugins different process-wide
ONNX Runtime ABIs.

The repository retains the existing CPU-only vendor packages for independent
consumers. They are not alternative providers for these conversion cores and
must not be installed alongside their conflicting CUDA-capable provider.

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

ONNX Runtime has no public C++ DLPack importer. The C++ CUDA plugin constructs
an `Ort::Value` directly over the leased device pointer with
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

The Python CUDA plugin uses a private capsule helper because
`OrtValue.from_dlpack` is ONNX Runtime's Python device-pointer importer. This
is an implementation detail of the ONNX adapter, not a public/shared DLPack
package or cross-framework ABI.

## Validation

The release-like pipeline under `docker/resolute/debian/` builds production
Debians without a GPU, driver, preinstalled CUDA, ONNX Runtime, or Python
ONNX Runtime. A pristine consumer installs the core and CPU plugins, builds
the unshipped test sources externally and runs them manually, installs only
the CUDA conversion plugins, verifies the core files are unchanged, and runs
the C++/Python CUDA unit and process-boundary tests on a GPU.

## License

Apache-2.0
