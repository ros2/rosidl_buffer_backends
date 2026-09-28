<!--
Copyright 2026 Open Source Robotics Foundation, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# cuda_buffer_rs

Rust access to ROS CUDA buffers, with typed `cuda-core` read and write handles.
The native backend owns GPU allocations and records synchronization events.

## Build from source

Requires Linux, Rust 1.89+, CUDA 13+, libclang, and a Buffer-enabled ROS 2 source
workspace with Rust message generation. Install `colcon-cargo`,
`colcon-ros-cargo`, and `cargo-ament-build` for Rust ROS packages.
Cargo downloads the pinned `cuda-core` 0.3.1 dependency from crates.io; no ROS
vendor package is required.

From the workspace root:

```bash
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --packages-up-to cuda_buffer_rs
source install/setup.bash
```

Add these dependencies to the application's `Cargo.toml`:

```toml
[dependencies]
cuda_buffer_rs = { version = "0.1", features = ["cuda-core"] }
cuda-core = "=0.3.1"
rclrs = "0.7"
ros-env = "=0.2.0"
```

Declare `cuda_buffer_rs`, `rclrs`, and `sensor_msgs` in `package.xml`.
Colcon resolves workspace ROS crates through the ament index; `cuda-core` and
other crates.io dependencies use Cargo's normal registry and cache. `ros-env`
exposes the generated message types.

The `cuda-core` feature enables the typed API shown below. Without it, the crate
provides the low-level native buffer and raw-pointer API.

## Usage

These snippets use an existing ROS `node`.
`produce_device_data` and `consume_device_data` represent your CUDA kernels,
queued on that stream.

### Publisher (direct write, zero-copy)

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
let mut output = from_output_buffer::<u8>(&mut image.data, &stream)?;
unsafe {
    produce_device_data(output.get_ptr(), output.len(), &stream);
}
publisher.publish(image)?;
```

Queue a kernel that writes the entire buffer on `stream`.

### Publisher (copy from an existing pointer)

With `image.data` already allocated, copy from an existing device allocation:

```rust
use cuda_buffer_rs::{from_output_buffer, to_buffer, CopyKind};

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
publisher.publish(image)?;
```

Use `CopyKind::HostToDevice` for a host pointer. Keep the source allocation valid
until the copy completes, and order its producer before the copy.

### Subscriber (read from a buffer, zero-copy)

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
            consume_device_data(input.get_ptr(), input.len(), &stream);
        }
    },
)?;
```

`input.get_ptr()` points to the received CUDA storage without copying. For CPU
input, `from_input_buffer` uploads the data to a temporary allocation owned by
the read handle.

CPU applications can keep `sensor_msgs::msg::Image` and its `Vec<u8>` payload.
For CPU-backed `msg::buffer::Image`, use `data.as_slice()` to borrow the pixels.

### Ownership and streams

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

`as_device_buffer()` and `as_device_buffer_mut()` borrow `cuda_core::DeviceBuffer`
views without copying. These methods are unsafe: callers must follow the
handle's stream contract and must not replace, resize, free, or take ownership
of the view. The view's normal allocator destructor must never run on the
backend's VMM allocation. Use `to_host_vec()` or `copy_from_host()` for checked,
synchronous host transfers.

## Tests

All test sources are under `test/`. Cargo runs the buffer API tests in
`unit.rs`, message/service tests in `message_compatibility.rs`, and the test
runner's argument and sequence-validation tests:

```bash
colcon test --packages-select cuda_buffer_rs --return-code-on-test-failure
colcon test-result --verbose
```

`colcon.pkg` enables `cuda-core`. GPU tests require a CUDA-capable device.
Python launch tests run separately from `colcon test`:

```bash
cuda_rs_pkg="$PWD/src/rosidl_buffer_backends/cuda_buffer_backend/cuda_buffer_rs"
cargo build --manifest-path "$cuda_rs_pkg/Cargo.toml" \
  --example cuda_buffer_rs_test_runner --features cuda-core \
  --target-dir "$cuda_rs_pkg/target"

launch_test "$cuda_rs_pkg/test/test_cuda_image_inter_pubsub_fastrtps_launch.py"
launch_test "$cuda_rs_pkg/test/test_cuda_image_intra_pubsub_fastrtps_launch.py"
launch_test "$cuda_rs_pkg/test/test_cuda_image_cpu_fallback_fastrtps_launch.py"
```

The launch files use one `cuda_buffer_rs_test_runner` executable, backed by
separate publisher and subscriber modules in `test/src/`. Its modes are
`publisher <subscriber-count>`, `subscriber <id> [cuda|cpu]`, and
`composed <subscriber-count>`. To use another build directory:

```bash
launch_test "$cuda_rs_pkg/test/test_cuda_image_inter_pubsub_fastrtps_launch.py" \
  test_runner:=/absolute/path/to/cuda_buffer_rs_test_runner
```

The three scenarios cover:

- Inter-process: one publisher and two GPU subscriber processes.
- Same-process: one executor hosts a publisher and three GPU subscriber nodes.
- CPU fallback: one publisher, one GPU subscriber, and one ordinary
  `sensor_msgs::msg::Image` subscriber in separate processes.

The same-process test does not exercise rclcpp's optimized intra-process
transport, which rclrs 0.7 does not expose.

Each scenario checks ten images for pixel values, metadata, CUDA backend
selection, and delivery order. GPU readers and writers use both default and
non-blocking streams. A delayed producer checks event ordering.
Sequence zero negotiates the backend before numbered samples are sent; each
subscriber acknowledges on its own topic.

The launch files select Fast DDS and use `EnableRmwIsolation` from
`launch_testing_ros`, so concurrent runs can share the fixed `test_cuda_image`
topic. Python checks success markers and requires each process to exit with
code zero. CI must run the three `launch_test` commands explicitly.

## Debian packaging

The tested Bloom 0.14.4 workflow rejects the `ament_cargo` build type, so the
Rust backend currently requires a source build. Installing `colcon-ros-cargo`
does not add Bloom support.
