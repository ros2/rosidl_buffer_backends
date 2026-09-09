# DLPack conversions

Framework-free conversions between `tensor_msgs/ExperimentalTensor` and
DLPack.

Everything device-specific lives in a storage plugin, and every plugin speaks
only DLPack. The core allocates message storage, hands out DLPack tensors over
it, and keeps that storage alive for as long as a consumer holds the tensor.
Framework adapters such as [`torch_conversions`](../torch_conversions/README.md)
sit on top and only translate between DLPack and their own tensor type, so no
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

`available_backends()` lists what the process can reach and
`default_backend()` reports what an unnamed call will use, which is the
installed plugin with the highest priority. Pass a backend name to
`allocate_tensor_msg` to override it for one message, or set
`ROSIDL_TENSOR_BACKEND` to pin the whole process. Naming a backend that is not
installed is an error rather than a silent fall back to host memory.

## C++

```bash
colcon build --merge-install --packages-up-to dlpack_conversions_cuda
```

```cpp
#include "dlpack_conversions/dlpack_conversions.hpp"

auto msg = dlpack_conversions::allocate_tensor_msg(
  {480, 640, 3}, DLDataType{kDLUInt, 8, 1}, "cuda");

// The lease keeps the storage mapped, so hold it while DLPack is in use.
auto managed = dlpack_conversions::from_output_tensor_msg(*msg, stream);
my_pipeline(managed.get()->dl_tensor);
publisher->publish(std::move(msg));
```

`from_input_tensor_msg` is the read-side counterpart. Both reject a message
whose shape, strides, and offset describe a view reaching outside its own
storage, so a malformed message cannot hand a framework an out-of-bounds
pointer.

To copy a tensor a framework already owns into message storage, use
`to_tensor_msg`. The copy is performed by the plugin that owns the memory,
which is how device-to-host and device-to-device copies stay out of the
adapters.

## Python

```bash
colcon build --merge-install --packages-up-to dlpack_conversions_py_cuda
```

```python
from dlpack_conversions import allocate_tensor_msg
from dlpack_conversions import from_output_tensor_msg
from dlpack_conversions import to_tensor_msg

msg = allocate_tensor_msg((480, 640, 3), (1, 8, 1), 'cuda')
capsule = from_output_tensor_msg(msg, stream)
```

The Python core exposes the same conversions and returns DLPack capsules,
which any array library consumes through `__dlpack__`. `to_tensor_msg` takes a
capsule and copies it into message storage, allocating on the capsule's own
device when no destination is given.

## Writing a storage plugin

A plugin declares the backend names and DLPack device types it serves, then
implements allocation, input and output acquisition, and `copy_to`. C++
plugins export through pluginlib; Python plugins register through an ament
index resource. Because the interface is DLPack and nothing else, a new
accelerator needs no change to the core or to any framework adapter.

## License

Apache-2.0
