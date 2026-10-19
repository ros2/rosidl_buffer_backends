# cuda_buffer_backend

CUDA buffer backend plugin for the ROS 2 Buffer system. Enables zero-copy GPU memory sharing between publishers and subscribers on the same host using CUDA VMM (Virtual Memory Management).

See the [design document](docs/cuda_buffer_backend_design.md) for architecture
and implementation details.

## Prerequisites

- A ROS 2 Rolling source workspace; see
  [Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
  for the canonical setup.
- CUDA Toolkit (>= 11.8) on the host.

## Build

After cloning this repo into your workspace's `src/` directory:

```bash
# Install system dependencies (CUDA toolkit, etc.).
rosdep install --from-paths src --ignore-src -y \
  --skip-keys "fastcdr rti-connext-dds-7.7.0 urdfdom_headers qt6-svg-dev"

# Build the CUDA backend.
colcon build --symlink-install --packages-up-to cuda_buffer_backend cuda_buffer_py
source install/setup.sh
```

## Test

```bash
colcon test --packages-select cuda_buffer cuda_buffer_py cuda_buffer_backend
colcon test-result --verbose
```

## Packages

| Package | Description |
|---|---|
| `cuda_buffer` | Core CUDA buffer implementation: memory pool, IPC manager, host endpoint manager, and user-facing `allocate_buffer` / `from_input_buffer` / `from_output_buffer` / `to_buffer` APIs |
| `cuda_buffer_py` | Python allocation, scoped read/write handles, and copying into open write handles for rclpy or standalone use |
| `cuda_buffer_backend` | Plugin registration via `pluginlib`, endpoint discovery, and descriptor serialization |
| `cuda_buffer_backend_msgs` | ROS 2 message definition for `CudaBufferDescriptor` |

## Usage

### Publisher (direct write, zero-copy)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "sensor_msgs/msg/image.hpp"

const size_t data_size = 640 * 480 * 3;

sensor_msgs::msg::Image msg;
msg.data = cuda_buffer_backend::allocate_buffer(data_size);
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

{
  cuda_buffer_backend::WriteHandle wh =
    cuda_buffer_backend::from_output_buffer(msg.data, stream);
  my_kernel<<<...>>>(wh.get_ptr(), ...);
}  // WriteHandle destructor records the write event on `stream`

publisher->publish(msg);
```

`allocate_buffer(count)` returns a `rosidl::Buffer<uint8_t>`; the caller
assigns it to whichever field the message schema uses.

### Publisher (copy from existing pointer)

Use `to_buffer` to copy bytes from an existing pointer (host or device) into
a buffer that was already allocated (e.g. via `allocate_buffer`). `to_buffer`
is a plain memcpy-through-a-WriteHandle and does **not** allocate.

```cpp
sensor_msgs::msg::Image msg;
msg.data = cuda_buffer_backend::allocate_buffer(data_size);
msg.height = 480;
msg.width = 640;
msg.encoding = "rgb8";
msg.step = 640 * 3;

{
  cuda_buffer_backend::WriteHandle wh =
    cuda_buffer_backend::from_output_buffer(msg.data, stream);

  // From a device pointer (D2D copy, default kind)
  cuda_buffer_backend::to_buffer(gpu_ptr, data_size, wh, stream);

  // Or from a host pointer (H2D copy)
  // cuda_buffer_backend::to_buffer(
  //   host_ptr, data_size, wh, stream, cudaMemcpyHostToDevice);
}  // wh destructor records the write event on `stream`

publisher->publish(msg);
```

### Subscriber (read from buffer, zero-copy)

```cpp
#include "cuda_buffer/cuda_buffer_api.hpp"

void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  cuda_buffer_backend::ReadHandle rh =
    cuda_buffer_backend::from_input_buffer(msg->data, stream);
  // ReadHandle constructor waits on publisher's write_event.
  // rh.get_ptr() returns `const uint8_t *` — the type system enforces read-only access.

  my_kernel<<<...>>>(rh.get_ptr(), ...);
}  // ReadHandle destructor records the read event for buffer lifetime tracking
```

### Auto-promoting non-CUDA buffers

`from_input_buffer` / `from_output_buffer` accept any `rosidl::Buffer<T>`, not
just CUDA-backed ones. If the source is a non-CUDA buffer (e.g. the CPU
fallback path), a new CUDA-backed `rosidl::Buffer<uint8_t>` is allocated on the
fly and the returned handle points at it. The handle owns the promoted buffer;
call `handle.get_promoted_buffer()` to retrieve it.

### `from_input_buffer` vs `from_output_buffer`

Two explicit functions surface the input/output intent at the call site:

```cpp
// Output path (publisher): requires a mutable buffer and returns WriteHandle.
cuda_buffer_backend::WriteHandle wh =
  cuda_buffer_backend::from_output_buffer(msg.data, stream);

// Input path (subscriber): accepts both const and mutable buffers and returns ReadHandle.
cuda_buffer_backend::ReadHandle rh =
  cuda_buffer_backend::from_input_buffer(msg->data, stream);
```

- `from_output_buffer` takes `rosidl::Buffer<T> &`; it acquires exclusive write
  access and can only be called once per buffer. A second call (or one after
  finalization) throws `CudaError`. `WriteHandle::get_ptr()` returns
  `uint8_t *`.
- `from_input_buffer` takes `const rosidl::Buffer<T> &`; it accepts both
  const and mutable arguments (const-ref binding). `ReadHandle::get_ptr()` returns
  `const uint8_t *` — the type system prevents subscribers from writing
  through the handle.

### Python (rclpy)

The `cuda_buffer_py` package exposes the same four core operations as C++:

| Operation | Responsibility |
| --- | --- |
| `CudaBuffer.allocate_buffer(size)` | Allocate uninitialized CUDA storage, without acquiring a producer handle. |
| `CudaBuffer.from_output_buffer(buffer, stream=None)` | Acquire the single producer write handle; promote non-CUDA output using its byte size, without copying contents. |
| `CudaBuffer.from_input_buffer(buffer, stream=None)` | Acquire a consumer read handle, promoting CPU input automatically. |
| `CudaBuffer.to_buffer(source, byte_count, handle, stream=None)` | Copy into an existing, open `CudaWriteHandle`. |

Standalone use requires no Torch installation. For a direct CUDA producer,
allocate device storage and write it on the same stream used by the producing
kernel:

```python
from cuda_buffer import CudaBuffer
from sensor_msgs.msg import Image

stream_ptr = get_current_cuda_stream_ptr()

msg = Image()
msg.height = 480
msg.width = 640
msg.encoding = 'rgb8'
msg.step = 640 * 3
msg.data = CudaBuffer.allocate_buffer(msg.height * msg.step)

with CudaBuffer.from_output_buffer(msg.data, stream_ptr) as write_handle:
    produce_device_data(
        write_handle.device_ptr,
        len(msg.data),
        stream_ptr,
    )

publisher.publish(msg)
```

Leaving the context records the producer event on `stream_ptr` and finalizes
the payload; it does not synchronize the stream. A second write handle cannot
be acquired for that buffer. Both handle types expose
`get_promoted_buffer() -> Optional[Buffer]`, matching C++. It returns the new CUDA
buffer when promotion occurred, or `None` for an already-CUDA argument. There is
no `.buffer` property. Retrieve the promoted buffer while the handle is open;
the returned object is the same Python `Buffer` retained by the handle and remains
valid after the handle closes. Repeated retrieval does not consume the promotion.
Closed handles reject pointer access and `get_promoted_buffer()` with `RuntimeError`.

The original CPU buffer is unchanged. Assign the promoted output back to the
message before publishing:

```python
with CudaBuffer.from_output_buffer(msg.data, stream_ptr) as write_handle:
    promoted = write_handle.get_promoted_buffer()
    if promoted is not None:
        msg.data = promoted
    produce_device_data(write_handle.device_ptr, len(msg.data), stream_ptr)

publisher.publish(msg)
```

Promotion is implemented in the Python bindings. They allocate a Python-owned
CUDA buffer before acquiring native handles and remember whether the original
argument required promotion. Existing CUDA buffers are used directly. Input
promotion copies CPU contents through an internal producer write scope, closes
that writer, and acquires a read handle. Output promotion acquires a write handle
without copying CPU contents. Both paths preserve the C++ single-write lifecycle.

The acquisition functions also accept `bytes`, `bytearray`, `memoryview`, and
iterables of byte values. Containers use the byte sequence produced by
`bytes(container)`, including the full byte count of typed arrays or views.
No temporary native CPU buffer is required. For new GPU payloads, use explicit
`allocate_buffer()` followed by write-handle acquisition.

For CPU-originated data, allocate storage, acquire a write handle, and use
`to_buffer()` to copy into it. The copy leaves the handle open for further GPU
writes in the same scope:

```python
from cuda_buffer import CudaBuffer
from sensor_msgs.msg import Image

stream_ptr = get_current_cuda_stream_ptr()

msg = Image()
msg.height = 480
msg.width = 640
msg.encoding = 'rgb8'
msg.step = 640 * 3
msg.data = CudaBuffer.allocate_buffer(len(host_image_bytes))
with CudaBuffer.from_output_buffer(msg.data, stream_ptr) as write_handle:
    CudaBuffer.to_buffer(host_image_bytes, len(host_image_bytes), write_handle, stream_ptr)
    # Optional additional GPU writes on stream_ptr before the handle closes.
    produce_device_data(write_handle.device_ptr, len(msg.data), stream_ptr)
publisher.publish(msg)
```

`source` accepts host `bytes`, `bytearray`, `memoryview`, or an iterable of byte
values, or an integer device pointer. The byte count is explicit and must be
nonnegative and no larger than the destination; host copies also validate the
source length. A smaller count copies only that prefix. Zero bytes is a no-op
on an open handle. Closed handles raise `RuntimeError`; invalid sizes, a null
device pointer for a nonempty copy, and mismatched streams raise `ValueError`.
For device pointers, the caller must supply a valid allocation of at least
`byte_count` bytes.

`to_buffer()` uses the handle's stream when `stream` is omitted. An explicit
stream must match the stream used to acquire the handle, so its completion
event covers the copy and subsequent writes. It does not allocate a destination,
acquire another handle, or close the supplied handle. Host copies synchronize
that stream before returning, keeping temporary host storage alive until the
copy completes; the caller may then change or release the source. For zero
initialization, copy `bytes(size)` through the same allocation/write/copy path.

Device-to-device copies are asynchronous. Make the source ready on the copy
stream and keep its storage alive until the copy completes. A CUDA read handle
on the same stream provides both ordering and allocation lifetime:

```python
destination = CudaBuffer.allocate_buffer(len(source_buffer))
with CudaBuffer.from_output_buffer(destination, stream_ptr) as write_handle:
    with CudaBuffer.from_input_buffer(source_buffer, stream_ptr) as read_handle:
        CudaBuffer.to_buffer(
            read_handle.device_ptr, len(source_buffer), write_handle, stream_ptr)
    produce_device_data(write_handle.device_ptr, len(destination), stream_ptr)
```

Python subscribers can acquire the same scoped CUDA read access as the C++
`from_input_buffer()` API. Pass the CUDA stream that will consume the data as
an integer pointer; CPU fallback data is promoted automatically:

```python
from cuda_buffer import CudaBuffer

def callback(msg):
    stream_ptr = get_current_cuda_stream_ptr()
    with CudaBuffer.from_input_buffer(msg.data, stream_ptr) as read_handle:
        consume_device_data(
            read_handle.device_ptr,
            len(msg.data),
            stream_ptr,
        )
```

For an existing CUDA-backed field, `from_input_buffer()` only enqueues a wait
for the producer event on the consumer stream. The handle keeps the source or
promoted CUDA buffer alive, and leaving the context records a read event so the
backend does not recycle the allocation before queued work completes. There is
no CPU conversion or stream synchronization in this CUDA-to-CUDA path. If no
stream is supplied, the backend's process-lifetime internal stream is used.
CPU fallback input is promoted with a host-to-device copy on the selected
stream, synchronized before returning so its host source can safely be released.
To keep or forward that CUDA copy after reading, save
`read_handle.get_promoted_buffer()` inside the context and retain the returned
`Buffer`. This does not acquire a producer handle or reopen the finalized payload.

Subscribers opt in to delivery through the CUDA backend with rclpy's
`acceptable_buffer_backends` argument. CPU remains an implicit fallback when
CUDA IPC is unavailable:

```python
subscription = node.create_subscription(
    Image,
    'image',
    callback,
    10,
    acceptable_buffer_backends='cuda',
)

def callback(msg):
    if getattr(msg.data, 'backend_type', 'cpu') == 'cuda':
        host_image_bytes = msg.data.to_bytes()
    else:
        host_image_bytes = bytes(msg.data)
```

## IPC Behavior

The RMW layer calls `on_discovering_endpoint()` for each subscriber to decide between zero-copy IPC and CPU fallback:

| Condition | Path |
|---|---|
| Same host, same GPU, same user | Zero-copy via CUDA VMM IPC |
| Different GPU, different user, different host, or VMM unavailable | CPU fallback via `to_cpu()` |

The publisher's pool checks a shared-memory refcount before recycling a block, ensuring all IPC subscribers have released their handles.

## License

Apache-2.0
