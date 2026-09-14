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
handle does not own the buffer. Python views retain their backing buffer, but
it must not be resized while a view exists.

Queue all producer writes on the conversion's execution stream before
publishing, and do not write to the published storage afterward. The CUDA
backend finalizes an outstanding writer when exporting the message or acquiring
a read handle, recording the event that readers wait on. Output values and I/O
bindings may remain alive during publication; destroying them is not required
to finalize the writer. Do not run inference again with a published message
still bound as an output.

The C++ examples publish by reference to keep the message alive while borrowed
views exist. Release those views before transferring message ownership with
`publish(std::move(msg))`. Python native values retain their backing buffer.

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

### Publisher

Create the stream and session once in the publisher. `env`, the serialized
`model`, and `input_value` belong to the application. The shape, dtype, and
binding names must match the model; the input must be ready on the session's
stream. These examples assume valid, nonempty messages.

```cpp
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

auto stream = onnxruntime_conversions::create_stream(env);
Ort::SessionOptions options;
onnxruntime_conversions::configure_session_options(options, stream);
Ort::Session session(env, model, model_size, options);

auto msg = onnxruntime_conversions::allocate_tensor_msg(
  {480, 640, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, stream);
auto output = onnxruntime_conversions::from_output_tensor_msg(*msg, stream);
Ort::IoBinding binding(session);
binding.BindInput("input", input_value);
binding.BindOutput("output", output.value());
session.Run(Ort::RunOptions{}, binding);
publisher->publish(*msg);
```

Allocation, conversion, and the session use the same stream. Inference writes
directly into message storage; no `to_tensor_msg()` copy is needed.

To borrow a caller-owned stream, replace only stream creation:

```cpp
auto stream = onnxruntime_conversions::borrow_stream(
  existing_stream, "cuda", /*device_id=*/0);
```

Keep the native stream owner alive until the session, views, and queued work
finish; the borrowed wrapper does not extend that ownership.

### Subscriber

Create a stream once in the subscriber and capture it in the callback. The
application function `consume()` must treat the input as read-only and use the
supplied stream, or a session configured with that stream.

```cpp
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

auto stream = onnxruntime_conversions::create_stream(env);
auto callback = [&stream](const onnxruntime_conversions::TensorMsg::ConstSharedPtr & received) {
    auto view = onnxruntime_conversions::from_input_tensor_msg(*received, stream);
    consume(view.value(), stream);
  };
```

Keep `env` and `stream` alive for the subscription and its queued work. Keep
the C++ view and message alive until `consume()` finishes using them.
An `Ort::Value` alone does not retain the C++ view's access handle.

### Existing tensors

Use `to_tensor_msg()` only when copying an already-existing ONNX Runtime tensor
into a message. Use the stream associated with its producer:

```cpp
auto outgoing = onnxruntime_conversions::to_tensor_msg(ort_value, stream);
publisher->publish(std::move(outgoing));
```

To reuse an existing message allocation with sufficient capacity:

```cpp
onnxruntime_conversions::to_tensor_msg(*preallocated, ort_value, stream);
publisher->publish(std::move(preallocated));
```

Copies complete before returning. Passing a stream here does not change the
producer session's stream; synchronize explicitly if they differ.

## Python

Install CPU support:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py \
  ros-$ROS_DISTRO-onnxruntime-conversions-py-cpu
```

Add CUDA support later:

```bash
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

### Publisher

Create the stream and session once in the publisher. `model` is the
application's model path or serialized model. The shape, dtype, and binding
names must match the model; `input_value` must be ready on the session's stream.

```python
import numpy as np
import onnxruntime as ort
from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import create_stream
from onnxruntime_conversions import from_output_tensor_msg
from onnxruntime_conversions import session_providers

stream = create_stream()
session = ort.InferenceSession(
    model, providers=session_providers(stream=stream))

msg = allocate_tensor_msg((480, 640, 3), np.float32, stream=stream)
output = from_output_tensor_msg(msg, stream)
binding = session.io_binding()
binding.bind_ortvalue_input('input', input_value)
binding.bind_ortvalue_output('output', output.value)
session.run_with_iobinding(binding)
publisher.publish(msg)
```

Allocation, conversion, and the session use the same stream. Inference writes
directly into message storage; no `to_tensor_msg()` copy is needed.

To borrow a caller-owned stream, replace only stream creation:

```python
from onnxruntime_conversions import borrow_stream

stream = borrow_stream(existing_stream, 'cuda', device_id=0)
```

The native handle is an integer. Keep its owner alive until the session, views,
and queued work finish; the borrowed wrapper does not extend that ownership.

### Subscriber

Create a stream once in the subscriber and use it in the callback. The
application function `consume()` must treat the input as read-only and use the
supplied stream, or a session configured with that stream.

```python
from onnxruntime_conversions import create_stream
from onnxruntime_conversions import from_input_tensor_msg

stream = create_stream()


def callback(received):
    view = from_input_tensor_msg(received, stream)
    consume(view.value, stream)
```

Keep `stream` alive for the subscription and its queued work.

Both conversion functions return an `OrtTensorView` whose `.value` is a native
`OrtValue`; empty buffers return `None`. Unlike C++, the Python native value
retains the storage and access handle through DLPack, even after the wrapper
is released.

The wrapper's `with` syntax is optional. It drops the wrapper's references on
exit, but does not release values or I/O bindings retained elsewhere, and does
not synchronize the stream. The examples need neither `with` for views nor
explicit `del` statements.

### Existing tensors

Use `to_tensor_msg()` only when copying an already-existing ONNX Runtime tensor
into a message. Use the stream associated with its producer:

```python
from onnxruntime_conversions import to_tensor_msg

outgoing = to_tensor_msg(ort_value, stream=stream)
publisher.publish(outgoing)
```

To reuse an existing message allocation with sufficient capacity:

```python
to_tensor_msg(preallocated, ort_value, stream=stream)
publisher.publish(preallocated)
```

Copies complete before returning. Passing a stream here does not change the
producer session's stream; synchronize explicitly if they differ.

## License

Apache-2.0
