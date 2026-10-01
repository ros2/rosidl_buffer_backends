<!--
Copyright 2026 Open Source Robotics Foundation, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# cuda_buffer_backend

CUDA buffer backend plugin for the ROS 2 Buffer system. Enables zero-copy GPU memory sharing between publishers and subscribers on the same host using CUDA VMM (Virtual Memory Management).

See the [design document](docs/cuda_buffer_backend_design.md) for architecture
and implementation details.

## Prerequisites

- A ROS 2 Rolling source workspace; see
  [Building ROS 2 on Ubuntu](https://docs.ros.org/en/rolling/Installation/Alternatives/Ubuntu-Development-Setup.html)
  for the canonical setup.
- CUDA Toolkit (>= 11.8) on the host.
- For Rust: Linux, Rust 1.89+, CUDA 13+, libclang, and Rust message generation
  in the Buffer-enabled ROS workspace. Install `colcon-cargo`,
  `colcon-ros-cargo`, and `cargo-ament-build` for Rust ROS packages.
  Cargo downloads `cuda-core` 0.3.1 from crates.io; no ROS vendor package is
  required.

## Build

After cloning this repo into your workspace's `src/` directory:

```bash
# Install system dependencies (CUDA toolkit, etc.).
rosdep install --from-paths src --ignore-src -y \
  --skip-keys "fastcdr rti-connext-dds-7.7.0 urdfdom_headers qt6-svg-dev"

# Build the CUDA backend and Rust bindings.
colcon build --symlink-install --packages-up-to cuda_buffer_backend cuda_buffer_rs
source install/setup.sh
```

The Rust backend currently requires a source build: the tested Bloom 0.14.4
workflow rejects the `ament_cargo` build type. Installing `colcon-ros-cargo`
does not add Bloom support. Omit `cuda_buffer_rs` from the build command if
Rust bindings are not needed.

## Test

Requires a CUDA-capable device. For Rust, install `rustfmt` for the active
toolchain before running `colcon test`:

```bash
# For an apt-managed toolchain:
sudo apt-get install rustfmt
# For a rustup-managed toolchain, use: rustup component add rustfmt

colcon test --packages-select cuda_buffer cuda_buffer_backend cuda_buffer_rs \
  --return-code-on-test-failure
colcon test-result --verbose
```

For `cuda_buffer_rs`, `colcon test` runs the unit tests, all three Python launch
tests, and the formatting check. `colcon build` builds the Rust test runner, so
an individual launch test can also run directly from the workspace root after
sourcing the workspace:

```bash
launch_test src/rosidl_buffer_backends/cuda_buffer_backend/cuda_buffer_rs/test/test_cuda_image_inter_pubsub_fastrtps_launch.py
```

For a custom build directory, pass
`test_runner:=/absolute/path/to/cuda_buffer_rs_test_runner`.

## Packages

| Package | Description |
|---|---|
| `cuda_buffer` | Core CUDA buffer implementation: memory pool, IPC manager, host endpoint manager, and user-facing `allocate_buffer` / `from_input_buffer` / `from_output_buffer` / `to_buffer` APIs |
| `cuda_buffer_backend` | Plugin registration via `pluginlib`, endpoint discovery, and descriptor serialization |
| `cuda_buffer_backend_msgs` | ROS 2 message definition for `CudaBufferDescriptor` |
| `cuda_buffer_rs` | Rust CUDA buffer allocation and scoped, typed read/write handles for rclrs publishers and subscribers |

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

### Rust (rclrs)

The `cuda_buffer_rs` package exposes the same scoped input/output model to
rclrs, with typed `cuda-core` read and write handles. Message fields and standalone
allocations use `rosidl_runtime_rs::Buffer<u8>` as their owning type.
`cuda_buffer_rs::allocate_buffer()` creates CUDA storage in that type;
`from_input_buffer()` and `from_output_buffer()` borrow it for CUDA access.
The native backend owns GPU allocations and records synchronization events.

The `cuda-core` feature is enabled by default. Use `default-features = false`
in a dependency declaration, or `--no-default-features` with Cargo, to build
only the low-level raw-pointer API.

These snippets use an existing ROS `node`. `produce_device_data` and
`consume_device_data` are application-defined kernel-launch helpers, not backend
APIs. They accept `&mut cuda_core::DeviceBuffer<u8>` and
`&cuda_core::DeviceBuffer<u8>`, respectively, plus the CUDA stream. Each helper
must enqueue all buffer access on that stream before returning, without taking
ownership of or replacing the borrowed buffer.

#### Publisher (direct write, zero-copy)

```rust
use cuda_buffer_rs::{allocate_buffer, from_output_buffer};
use cuda_core::CudaContext;
use ros_env::sensor_msgs::msg::buffer::Image;

let context = CudaContext::new(0)?;
let stream = context.new_stream()?;
let publisher = node.create_publisher::<Image>("image")?;
let mut image = Image {
    height: 480,
    width: 640,
    encoding: "rgb8".into(),
    step: 640 * 3,
    data: allocate_buffer(640 * 480 * 3)?,
    ..Default::default()
};
{
    let mut output = from_output_buffer::<u8>(&mut image.data, &stream)?;
    unsafe {
        produce_device_data(output.as_device_buffer(), &stream);
    }
} // Handle cleanup records the write event on this stream.
publisher.publish(image)?;
```

Queue a kernel that writes the entire buffer on `stream`.

#### Publisher (copy from an existing pointer)

With `image.data` already allocated, copy from an existing device allocation:

```rust
use cuda_buffer_rs::{from_output_buffer, to_buffer, CopyKind};

{
    let mut output = from_output_buffer::<u8>(&mut image.data, &stream)?;
    unsafe {
        to_buffer(
            gpu_ptr,
            byte_count,
            &mut output,
            &stream,
            CopyKind::DeviceToDevice,
        )?;
    }
}
publisher.publish(image)?;
```

Use `CopyKind::HostToDevice` for a host pointer. Keep the source allocation valid
until the copy completes, and order its producer before the copy.

#### Subscriber (read from a buffer, zero-copy)

```rust
use cuda_buffer_rs::from_input_buffer;
use cuda_core::CudaContext;
use rclrs::SubscriptionOptions;
use ros_env::sensor_msgs::msg::buffer::Image;

let context = CudaContext::new(0)?;
let stream = context.new_stream()?;
let subscription = node.create_subscription::<Image, _>(
    SubscriptionOptions::new("image").acceptable_buffer_backends("cuda"),
    move |image: Image| {
        let input = from_input_buffer::<u8>(&image.data, &stream).unwrap();
        unsafe {
            consume_device_data(input.as_device_buffer(), &stream);
        }
    },
)?;
```

`input.as_device_buffer()` borrows the received CUDA storage without copying.
For CPU input, `from_input_buffer` uploads the data to a temporary allocation
owned by the read handle.

CPU applications can keep `sensor_msgs::msg::Image` and its `Vec<u8>` payload.
For CPU-backed `msg::buffer::Image`, use `data.as_slice()` to borrow the pixels.

#### Rust kernel interoperability

[cuda-oxide's typed launch API](https://nvlabs.github.io/cuda-oxide/gpu-programming/launching-kernels.html)
accepts `&DeviceBuffer<T>` for read-only slice arguments and
`&mut DeviceBuffer<T>` for writable slice arguments. The helpers above can pass
the borrowed views directly to that API; no raw-pointer extraction or buffer
copy is needed. The launcher marshals the device address and element count.
Use a cuda-oxide release whose `cuda-core` dependency resolves to the same
crate version and source as this backend (currently pinned to `0.3.1`).

[cuTile Rust](https://github.com/NVlabs/cutile-rs#quick-start) shares the
`cuda-core` runtime, but its high-level launch API uses tensors and partitioned
outputs, not bare `DeviceBuffer` borrows. This backend does not yet provide a
cuTile tensor adapter. Such an adapter must preserve backend ownership and
stream ordering; an API that takes ownership of a buffer cannot consume these
borrowed views directly.

Use `get_ptr()` only when a lower-level CUDA API requires a raw device pointer.

#### Ownership and streams

- CUDA-backed input and output reuse the backend allocation. CPU input is copied
  to temporary CUDA storage; CPU output is replaced with uninitialized CUDA
  storage of the same byte length.
- Read handles borrow their source. Write handles permit one initialization
  phase; queue all writes before publishing or reading the buffer.
- Handle cleanup records native access events and releases its retained stream.
  The backend frees or recycles the allocation after outstanding GPU work.
- Queue kernels and memory operations on the handle's stream. A
  `cuda_core::CudaStream` default stream means CUDA stream 0. The low-level
  `cuda_buffer_rs::CudaStream::INTERNAL` instead selects the backend's internal
  stream, resolved to its actual pointer before calling the C API.

`as_device_buffer()` borrows a `cuda_core::DeviceBuffer` view without copying:
read handles return `&DeviceBuffer<T>`, and write handles return
`&mut DeviceBuffer<T>`. Both methods are unsafe: callers must follow the
handle's stream contract and must not replace, resize, free, or take ownership
of the view. The view's normal allocator destructor must never run on the
backend's VMM allocation. Use `to_host_vec()` or `copy_from_host()` for checked,
synchronous host transfers.

## IPC Behavior

The RMW layer calls `on_discovering_endpoint()` for each subscriber to decide between zero-copy IPC and CPU fallback:

| Condition | Path |
|---|---|
| Same host, same GPU, same user | Zero-copy via CUDA VMM IPC |
| Different GPU, different user, different host, or VMM unavailable | CPU fallback via `to_cpu()` |

The publisher's pool checks a shared-memory refcount before recycling a block, ensuring all IPC subscribers have released their handles.

## License

Apache-2.0
