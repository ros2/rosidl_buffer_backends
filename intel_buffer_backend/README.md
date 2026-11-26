# Intel Buffer backend

Intel buffer backend plugin implementation of the ROS 2 rosidl buffer system for Intel Core Ultra architecture.
It enables zero-copy memory sharing across devices and across
processes using 3 transport paths:

1. Intel Level Zero USM
2. DMA-BUF
3. External Memory mapping extension using Intel Level Zero

## Build

Supported platform: Ubuntu 26.04 (amd64) with ROS 2 Lyrical or Rolling
([installation](https://docs.ros.org/en/rolling/Installing.html)), on an Intel
Core Ultra system with an integrated GPU.

All dependencies are installed by `rosdep`. From the workspace that has this
repository cloned under `src/`, install them and build:

```bash
cd ~/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep update
rosdep install --from-paths src/rosidl_buffer_backends/intel_buffer_backend --ignore-src -y
colcon build --symlink-install --packages-up-to intel_buffer_backend
source install/setup.bash
```

`rosdep` installs, among the ROS packages:

| Package | Used for |
|---|---|
| `libze-dev` | Level Zero headers and loader (required to build `intel_memory_core`) |
| `libze-intel-gpu1`, `intel-opencl-icd` | Intel GPU user-space drivers (Level Zero USM transport, GPU DMA-BUF import) |
| `opencl-headers`, `ocl-icd-opencl-dev` | GPU zero-copy import of DMA-BUFs (optional at build time) |
| `gtest` | `intel_memory_core` unit tests |

`intel_memory_core` stays a plain CMake library usable without ROS. Its
`package.xml` exists only so `rosdep`, `colcon` and `bloom` can resolve its
dependencies.

### Newer Intel hardware

The Ubuntu archive's GPU driver can lag behind the newest platforms. If the
Level Zero probe doesn't find the GPU (see [Verify](#verify) below), install the
latest compute-runtime from Intel's graphics PPA, as described in the
[Intel client GPU guide](https://dgpu-docs.intel.com/driver/client/overview.html),
then restart the nodes; no rebuild is needed:

```bash
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt update && sudo apt upgrade -y   # upgrades libze1, libze-intel-gpu1, intel-opencl-icd
```

NPU inference in the sample pipeline also needs the NPU driver
(`intel-level-zero-npu`, from the
[linux-npu-driver releases](https://github.com/intel/linux-npu-driver/releases)).
The buffer backend itself doesn't use the NPU driver.

## Test

```bash
colcon test --packages-select intel_memory_core intel_buffer intel_buffer_backend intel_buffer_backend_msgs
colcon test-result --verbose
```

Complete the [DMA-BUF setup](#dma-buf-setup-required) first. Tests for transports
the machine can't provide are skipped.

## Verify

The pool logs which transport it selected when the first node using it starts:

| Log line | Meaning |
|---|---|
| `Level Zero USM (device ordinal N); zero-copy across CPU/iGPU/NPU ...` | Level Zero GPU driver found; L0 USM transport active |
| `Using DMA-BUF transport for cross-process zero-copy` | No usable Level Zero GPU driver (or `INTEL_MEMORY_BUFFER_TYPE=dmabuf`); DMA-BUF transport active |
| `DMA-BUF unavailable: open(/dev/dma_heap/system, O_RDWR) failed ...` | The [DMA-BUF setup](#dma-buf-setup-required) is missing |
| `Neither Level Zero nor DMA-BUF available; ...` | No zero-copy transport; messages are sent as regular CPU copies |

## Packages

| Package | Description |
|---|---|
| `intel_memory_core` | USM/DMA-BUF memory pool, IPC managers, and the Level Zero external-memmap extension helpers shared by the other packages |
| `intel_buffer` | Core `IntelBufferImpl` storage plus user-facing `allocate_buffer` / `wrap_dmabuf` / `wrap_usm_ptr` / `wrap_external_memmap` / `ExternalMemMapRing` APIs |
| `intel_buffer_backend` | Plugin registration via `pluginlib`, endpoint discovery, and descriptor serialization |
| `intel_buffer_backend_msgs` | ROS 2 message definitions for `IntelBufferDescriptor` (and its `L0UsmDescriptor` / `DmaBufDescriptor` / `ExternalMemMapDescriptor` sub-messages) |

## Usage

Pick the publisher-side call based on where the frame data physically lives
*before* you publish. Getting this right is what makes the difference between
"zero-copy backend" and "zero-copy pipeline": the API will happily let you
`allocate_buffer()` + `memcpy()` every frame, but that defeats the point if
your data was already sitting in device-visible memory.

### Publisher — depth frame decoded on the CPU (one unavoidable copy)

A software depth decoder (e.g. an RVL or point-cloud codec running on the
host) produces its output in a plain `std::vector`/`cv::Mat` in CPU heap.
There is no way around one copy here, because the bytes don't yet live in
memory any device can see — `allocate_buffer()` gets a USM destination and
`write_to_buffer()` performs that single, unavoidable transfer:

```cpp
#include "intel_buffer/intel_buffer_api.hpp"
#include "sensor_msgs/msg/image.hpp"

const size_t frame_bytes = 640 * 480 * sizeof(uint16_t);  // 16-bit depth map

sensor_msgs::msg::Image msg;
msg.data = intel_buffer_backend::allocate_buffer(frame_bytes);
msg.height = 480;
msg.width = 640;
msg.encoding = "16UC1";
msg.step = 640 * sizeof(uint16_t);

intel_buffer_backend::write_to_buffer(msg.data, decoder.output(), frame_bytes);

publisher->publish(msg);
```

### Publisher — IPU7 camera frame already in a DMA-BUF (zero-copy)

`libcamhal`/V4L2 hands back a `dma_buf` fd that already points at the pages
the ISP wrote the frame into. `wrap_dmabuf()` does not touch the pixel
data at all — it only records the fd so the descriptor can be shared.
Subscribers read the camera's buffer in place, so give it back to the driver
only once `try_reclaim_dmabuf()` reports that no subscriber still holds it:

```cpp
int dmabuf_fd = camera.dequeue_buffer();  // VIDIOC_DQBUF + EXPBUF
msg.data = intel_buffer_backend::wrap_dmabuf(dmabuf_fd, frame_bytes, /*owns_fd=*/false);
publisher->publish(msg);

// Later, e.g. before each dequeue; retry while it returns false.
if (intel_buffer_backend::try_reclaim_dmabuf(dmabuf_fd)) {
  camera.requeue_buffer(dmabuf_fd);       // VIDIOC_QBUF
}
```

### Publisher — NPU inference output already in USM (zero-copy)

A pre-processing or detection model running on the NPU (via OpenVINO/Level
Zero) writes its output tensor directly into a `zeMemAllocHost()` region.
`wrap_usm_ptr()` publishes that same allocation as-is — nothing is staged
back through the CPU first. The backend can't free the pointer (it doesn't
know the allocator), so pass a deleter to hand it over; it runs once the
buffer and every message holding it are gone:

```cpp
void * tensor_ptr = npu_context.malloc_host<uint8_t>(tensor_bytes);
npu_model.infer_into(tensor_ptr);  // NPU writes the output tensor in place
msg.data = intel_buffer_backend::wrap_usm_ptr(tensor_ptr, tensor_bytes,
  [&npu_context](void * p) {npu_context.free(p);});
publisher->publish(msg);
```

### Publisher — high-rate producer, pool-free zero-copy

A node publishing at high frame rate (e.g. an IMU-synced stereo pair) that
wants to skip pool allocation entirely can own its memory outright with
`ExternalMemMapRing`. It creates a small rotation of page-aligned SHM regions
once at startup and maps each to the device a single time; every subsequent
frame costs one write plus a `wrap()` call:

```cpp
// startup
intel_buffer_backend::ExternalMemMapRing ring;
ring.create("/stereo_left_frame", frame_bytes, /*slots=*/4);

// per frame
const size_t slot = ring.advance();
render_directly_into(ring.data(slot), frame_bytes);
msg.data = ring.wrap(slot, frame_bytes);
publisher->publish(msg);
```

Ring depth is the only guard against the producer overwriting a slot a
subscriber is still reading — size it to outlast your slowest consumer.

### Subscriber — consume the frame in place

The subscriber never receives a copy of the message payload; `get_usm_ptr()`
resolves to the exact allocation the publisher wrote into (or wrapped), so an
iGPU or NPU kernel launched from the callback reads from that same address:

```cpp
#include "intel_buffer/intel_buffer_api.hpp"

void on_frame(const sensor_msgs::msg::Image::SharedPtr msg) {
  const uint8_t * frame = intel_buffer_backend::get_usm_ptr(msg->data);
  // No staging copy happened to get `frame` here — see below.
  run_inference_on_npu(frame, msg->data.size());
}
```

### Why this is *true* zero-copy across CPU, iGPU, and NPU

A conventional GPU/accelerator pipeline still pays for at least one staging
copy per hop: `malloc()` on the host, then an explicit
`cudaMemcpy`/`clEnqueueWriteBuffer`-style upload before the device can touch
the data, and a matching download before the CPU can read the result back.

This backend avoids that staging step entirely, on both ends of the
transport:

- **Level Zero USM (`wrap_usm_ptr`)** — `zeMemAllocHost()` memory is mapped
  into one address space that the CPU, the integrated GPU, and the NPU all
  resolve through their shared page tables. The pointer a publisher hands to
  `wrap_usm_ptr()` is the *same* pointer an NPU inference call or an iGPU
  kernel dereferences — there is no host-to-device or device-to-host copy at
  any point, because there is only one copy of the data to begin with.
- **DMA-BUF (`wrap_dmabuf`)** — the fd exported by the ISP/camera driver
  references physical pages directly importable by the `xe` (iGPU),
  `intel_ipu7`, and `intel_vpu` (NPU) kernel drivers. Publishing the fd (via
  IPC, not the bytes) lets a subscriber import the identical pages into its
  own device context.
- **External memmap (`ExternalMemMapRing`)** — the ring's POSIX SHM regions
  are mapped to the device once at startup, not once per frame, so steady-
  state publishing is a plain memory write followed by handing over a
  pointer + region name — no pool, no allocation, no fd handoff, no copy.

In all three paths, "zero-copy" means what it says: the number of times the
frame's bytes are duplicated, from producer write to consumer read across
CPU/iGPU/NPU, is zero. The only memcpy anywhere in this backend is the one in
`write_to_buffer()`, and that exists solely to bring data *into* shared memory
from a source that never had device visibility to begin with (e.g., a
software decoder or file read).

## IPC Behavior

The RMW layer calls `on_discovering_endpoint()` for each subscriber to decide between zero-copy IPC and CPU fallback:

| Condition | Path |
|---|---|
| Same host, both endpoints advertise `transport=extmemmap` and `extmemmap=1` | Zero-copy via the external-memmap transport (no pool required on either side) |
| Same host, matching pool transport (`l0usm` or `dmabuf`) | Zero-copy via Level Zero USM IPC or DMA-BUF fd export |
| Different transport, different host, or pool unavailable | CPU fallback via `to_cpu()` |

"Same host" means both endpoints advertise the same `host=` value, which is the
kernel boot ID (`/proc/sys/kernel/random/boot_id`). A peer that sends no `host=`
value is treated as being on a different host.

## DMA-BUF setup (required)

The buffer backend allocates every shared frame buffer from the kernel
**dma-heap** subsystem (`/dev/dma_heap/system`). This is now the only
fd-sharing path — the previous `memfd_create()` fallback has been removed, so
the dma-heap must be present and accessible or the pool reports *"DMA-BUF
unavailable"* and no frames are produced.

The dma_buf fd returned by the heap is kernel reference counted, mmap'able for
CPU access, and importable by the iGPU (`xe`), IPU (`intel_ipu7`) and NPU
(`intel_vpu`) drivers — so the same physical pages are shared zero-copy across
processes and across devices.

By default `/dev/dma_heap/system` is root-only. Grant the `render` group access
via the bundled udev rule and add your user to that group:

```bash
# 1. Install the udev rule (grants the 'render' group RW on the system heap)
sudo cp ./udev/60-intel-memory-dmabuf.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=dma_heap

# 2. Add your user to the 'render' group (log out / back in to take effect)
sudo usermod -aG render $USER

# 3. Verify: the node should be group 'render', mode crw-rw----
ls -l /dev/dma_heap/system
```

> The `render` group is reused deliberately — it already gates the iGPU render
> node (`/dev/dri/renderD128`), so a user who can run GPU inference already has
> the right group. Adjust the `GROUP=` field in the rule if your site uses a
> different group (e.g. `video`).

## Run the sample ROS2 pipeline

The sample ROS2 workload to test the intel buffer backend is located [here](./test_intel_buffer_backend/README.md)

## License

Apache-2.0
