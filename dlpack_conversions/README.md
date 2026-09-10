# DLPack conversions

Framework-free conversions between `tensor_msgs/ExperimentalTensor` and
DLPack.

This is the core that the framework adapters are built on, not the entry point
for application code. If you work in PyTorch or ONNX Runtime, depend on the
adapter for your framework and let it call in here:

| Framework | C++ | Python |
| --- | --- | --- |
| PyTorch | [`torch_conversions`](../torch_conversions/README.md) | `torch_conversions_py` |
| ONNX Runtime | `onnxruntime_conversions` | `onnxruntime_conversions_py` |

Reach for this package directly in two cases only: you are writing an adapter
or a storage plugin, or your consumer already speaks DLPack and has no
framework to adapt, as with NumPy, CuPy, or JAX through `__dlpack__`.

## How it fits together

Everything device-specific lives in a storage plugin, and every plugin speaks
only DLPack. The core allocates message storage, hands out DLPack tensors over
it, and keeps that storage alive for as long as a consumer holds the tensor.
Adapters only translate between DLPack and their own tensor type, so no
adapter contains device code and no framework decides where memory lands.

That split is what makes accelerators installable after the fact. A consumer
built against the core and the host plugin picks up CUDA storage as soon as
`dlpack_conversions_cuda` is installed, with no rebuild, because the plugin is
discovered at runtime through pluginlib in C++ and the ament index in Python.

| Device | C++ plugin | Python plugin |
| --- | --- | --- |
| Host | `dlpack_conversions_cpu` | `dlpack_conversions_py_cpu` |
| CUDA | `dlpack_conversions_cuda` | `dlpack_conversions_py_cuda` |

## Choosing a backend

This applies however you reach the core, through an adapter or directly.

`available_backends()` lists what the process can reach and
`default_backend()` reports what an unnamed call will use, which is the
installed plugin with the highest priority. Pass a backend name to
`allocate_tensor_msg` to override it for one message, or set
`ROSIDL_TENSOR_BACKEND` to pin the whole process. Naming a backend that is not
installed is an error rather than a silent fall back to host memory.

## Core API

For adapter and plugin authors, and for consumers already working in DLPack.

```cpp
#include "dlpack_conversions/dlpack_conversions.hpp"

auto msg = dlpack_conversions::allocate_tensor_msg(
  {480, 640, 3}, DLDataType{kDLUInt, 8, 1}, "cuda");

// The lease keeps the storage mapped, so hold it while DLPack is in use.
auto managed = dlpack_conversions::from_output_tensor_msg(*msg, stream);
my_pipeline(managed.get()->dl_tensor);
```

The Python core mirrors this and returns DLPack capsules, which any array
library consumes through `__dlpack__`.

```python
from dlpack_conversions import allocate_tensor_msg
from dlpack_conversions import from_output_tensor_msg

msg = allocate_tensor_msg((480, 640, 3), (1, 8, 1), 'cuda')
capsule = from_output_tensor_msg(msg, stream)
```

`from_input_tensor_msg` is the read-side counterpart. Both reject a message
whose shape, strides, and offset describe a view reaching outside its own
storage, so a malformed message cannot hand a framework an out-of-bounds
pointer.

To copy a tensor a caller already owns into message storage, use
`to_tensor_msg`. The copy is performed by the plugin that owns the memory,
which is how device-to-host and device-to-device copies stay out of the
adapters. Given a capsule and no destination, the Python core allocates on the
capsule's own device.

## Writing a storage plugin

A plugin declares the backend names and DLPack device types it serves, then
implements allocation, input and output acquisition, and `copy_to`. C++
plugins export through pluginlib; Python plugins register through an ament
index resource. Because the interface is DLPack and nothing else, a new
accelerator needs no change to the core or to any framework adapter.

## License

Apache-2.0
