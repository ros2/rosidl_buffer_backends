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
colcon build --symlink-install --packages-up-to cuda_buffer_backend
source install/setup.sh
```

## Test

```bash
colcon test --packages-select cuda_buffer cuda_buffer_backend
colcon test-result --verbose
```

## Packages

| Package | Description |
|---|---|
| `cuda_buffer` | Core CUDA buffer implementation: memory pool, IPC manager, host endpoint manager, and user-facing `allocate_buffer` / `from_input_buffer` / `from_output_buffer` / `to_buffer` APIs |
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

## IPC Behavior

The RMW layer calls `on_discovering_endpoint()` for each subscriber to decide between zero-copy IPC and CPU fallback:

| Condition | Path |
|---|---|
| Same host, same GPU, same user | Zero-copy via CUDA VMM IPC |
| Different GPU, different user, different host, or VMM unavailable | CPU fallback via `to_cpu()` |

The publisher's pool checks a shared-memory refcount before recycling a block, ensuring all IPC subscribers have released their handles.

## License

Apache-2.0
