# ONNX Runtime conversions

Conversions between `tensor_msgs/msg/ExperimentalTensor` and ONNX Runtime
tensors. The public packages are framework-specific and device-neutral:

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `onnxruntime_conversions` | `onnxruntime_conversions_py` |
| CPU implementation | `onnxruntime_conversions_cpu` | `onnxruntime_conversions_py_cpu` |
| CUDA implementation | `onnxruntime_conversions_cuda` | `onnxruntime_conversions_py_cuda` |

Install CPU and accelerator plugins as separate Debian packages without
rebuilding the conversion core. Plugins create device-appropriate streams;
applications can also borrow their own native streams. C++ discovers plugins
on first registry use; Python discovers them when the module is imported.
Restart the application after installing a plugin or provider and source the
ROS environment before starting it.

Default allocation and session setup use the available plugin with the highest
priority: CPU is 0 and CUDA is 100. Set `ROSIDL_TENSOR_BACKEND=cpu` or `cuda` to
override that default, or pass a backend explicitly. Views use the message's
storage backend; copies into a new message use the source tensor's device
unless the Python caller specifies a destination backend. Use `device_id` when
allocating on a particular device; copies into new messages preserve the source
device index. Additional accelerators can supply their own plugins and
compatible framework providers.

## ONNX Runtime provider boundary

All four providers use ONNX Runtime 1.26.0. The accelerator providers require
CUDA 12 and cuDNN 9.

The C++ core depends on `onnxruntime_core_vendor`; the Python core depends on
`python_onnxruntime_vendor`. Both use CPU-only distributions and require no
accelerator runtime or driver.

The CUDA plugins add `onnxruntime_cuda_vendor` and
`python_onnxruntime_cuda_vendor`, respectively. These depend on the CPU
providers and install matching CUDA-capable distributions in separate
directories. ROS environment hooks select the CUDA provider for newly started
processes after the environment is sourced again. Both plugins then share that
provider; CPU execution remains available.

Each process uses one ONNX Runtime distribution per language. CPU and
accelerator providers must have matching versions and compatible exported
symbols.

## Storage lifetime

C++ views borrow message storage on both CPU and CUDA: keep the message alive
and do not replace or resize its buffer while a view exists. A CUDA access
handle does not own the buffer. Release output views before publishing so their
write handles record completion on the execution stream. Python views retain
their backing buffer, but it must not be resized while a view exists. Release
output values and their I/O bindings before publishing.

## Execution streams

`create_stream()` selects a conversion plugin using the same backend argument,
environment override, and priorities as allocation. The CPU plugin returns an
empty stream; accelerator plugins create an ONNX Runtime-owned stream. The
handle carries its backend and device index so allocation and session setup
can use the same selection on either device.

Pass the `Stream` object directly to allocation, session setup, views, and
copies. Its backend and device accessors are for inspection; use `handle()` in
C++ or `handle` in Python when a native API needs the underlying handle.

Bind the execution stream when creating the session, using
`configure_session_options()` in C++ or `session_providers()` in Python. Pass
the same stream to conversions and subsequent operations, or explicitly
synchronize between streams. Passing a different stream to a conversion does
not change an existing session's stream.

The raw-handle APIs accept `nullptr` in C++ or `None` in Python for CPU session
setup. CUDA session setup requires a native handle; Python CUDA views and
copies accept a stream integer (`0` selects the legacy default). C++ views and
copies use the legacy default when the stream is omitted.

For a caller-owned stream, use `borrow_stream(native_handle, backend, device_id)`.
It does not create, replace, or destroy the native stream. The existing raw-handle
session and conversion APIs remain available. The CUDA plugin implements stream
creation through a registered `OrtEpDevice::CreateSyncStream()`; applications do
not need CUDA headers or their own provider-registration code.

Keep the stream owner alive until the session is destroyed, all views are
released, and queued work completes. In C++, keep `Ort::Env` alive longer than
its owned streams. Copies into messages complete before returning; zero-copy
views remain asynchronous.

Select the buffer's CUDA device before conversion and keep it selected when
releasing views. The existing CUDA buffer pool supports one allocation device
per process; choose that device before the first allocation. A request for a
different device fails instead of returning storage on the wrong GPU.

Copies between different GPU indices are rejected. Transfer the value
explicitly before copying it into storage on another device.

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

For a CPU-only build, restrict rosdep to `tensor_msgs`,
`onnxruntime_vendor/onnxruntime_core_vendor`, and the C++ core/CPU plugin
directories, then build only `onnxruntime_conversions_cpu`.

Create a stream, configure inference, and bind message storage without copying.
`env`, `model`, and `input_value` belong to the application; tensor shapes and
binding names must match the model.

```cpp
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

auto stream = onnxruntime_conversions::create_stream(env);
Ort::SessionOptions options;
onnxruntime_conversions::configure_session_options(options, stream);
Ort::Session session(env, model, model_size, options);

auto msg = onnxruntime_conversions::allocate_tensor_msg(
  {480, 640, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, stream);
{
  auto output = onnxruntime_conversions::from_output_tensor_msg(*msg, stream);
  Ort::IoBinding binding(session);
  binding.BindInput("input", input_value);
  binding.BindOutput("output", output.value());
  session.Run(Ort::RunOptions{}, binding);
}
publisher->publish(std::move(*msg));
```

This code runs on CPU or an installed accelerator. To use an existing stream,
replace only its creation with:

```cpp
auto stream = onnxruntime_conversions::borrow_stream(
  existing_stream, "cuda", /*device_id=*/0);
```

For received messages and existing values:

```cpp
auto input = onnxruntime_conversions::from_input_tensor_msg(received, stream);
my_pipeline(input.value());

auto outgoing = onnxruntime_conversions::to_tensor_msg(ort_value, stream);
onnxruntime_conversions::to_tensor_msg(*preallocated, ort_value, stream);
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

For a Python CPU-only build, restrict rosdep to `tensor_msgs`,
`onnxruntime_vendor/python_onnxruntime_vendor`, and the Python core/CPU plugin
directories, then build only `onnxruntime_conversions_py_cpu`.

```python
import numpy as np
import onnxruntime as ort
from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import create_stream
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg
from onnxruntime_conversions import session_providers
from onnxruntime_conversions import to_tensor_msg

stream = create_stream()
session = ort.InferenceSession(
    model, providers=session_providers(stream=stream))

msg = allocate_tensor_msg(
    (480, 640, 3), np.float32, stream=stream)
with from_output_tensor_msg(msg, stream) as output:
    binding = session.io_binding()
    binding.bind_ortvalue_input('input', input_value)
    binding.bind_ortvalue_output('output', output)
    session.run_with_iobinding(binding)
    del binding, output

with from_input_tensor_msg(msg, stream) as value:
    consume(value)

outgoing = to_tensor_msg(ort_value, stream=stream)
```

For a caller-owned stream, replace `create_stream()` with
`borrow_stream(existing_stream, 'cuda', device_id=0)`. The native handle is an
integer. Keep its owner alive for the full session lifetime; the borrowed
wrapper does not extend that ownership.

Python plugins use `OrtValue.from_dlpack` with a private capsule helper that
retains the storage and its access handle.

## License

Apache-2.0
