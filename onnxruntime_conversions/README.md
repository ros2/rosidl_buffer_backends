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

## Tensor layout

Both the C++ and Python APIs require contiguous, row-major tensor layouts on
CPU and CUDA. An empty message `strides` field implies this layout. Explicit
strides must exactly match the contiguous strides computed from `shape`, even
for dimensions of size one. Transposed or sliced layouts with other strides
are rejected; make the tensor contiguous before creating the message.

A nonzero `byte_offset` is supported when the entire contiguous tensor fits
within the message's allocated storage.

## Execution streams

`create_stream()` returns an empty CPU stream or an ONNX Runtime-owned
accelerator stream. Use `borrow_stream()` to wrap a caller-owned stream.

Pass the same `Stream` to allocation, views, and copies. Bind it to the session
using `configure_session_options()` in C++ or `session_providers()` in Python.
Passing another stream to a conversion does not change the session's stream;
synchronize explicitly when using different streams.

Keep the stream owner alive until sessions and views are released and queued
work finishes. In C++, `Ort::Env` must outlive its owned streams. Copies complete
before returning; zero-copy views remain asynchronous.

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

Copies complete before returning. Passing a stream here does not change the
producer session's stream; synchronize explicitly if they differ.

## License

Apache-2.0
