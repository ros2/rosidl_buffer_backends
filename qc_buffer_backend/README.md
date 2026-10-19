# qc_buffer_backend

A `rosidl::Buffer<T>` storage backend for Qualcomm platforms that delivers message payloads with **HTP–CPU zero-copy** in ROS2 pipeline.

Payloads are allocated in Qualcomm dma-buf memory. This memory is directly readable/writable by the CPU and, through its dma-buf fd, can be shared with the HTP(Qualcomm NPU) accelerator so it reads the exact same physical pages — no copy. When zero-copy is not possible (cross-device, or not QC device), the backend transparently falls back to the default CPU serialization path, so functionality is always correct.

The QC backend has no compile-time dependency on any Qualcomm SDK. The only dependency is `libcdsprpc.so`, and it is resolved entirely at runtime via dlopen. This means the package can be built on any standard Linux host without any Qualcomm-specific toolchain installed. On Qualcomm platforms, libcdsprpc.so ships as part of the board support package and is available by default on virtually all Qualcomm Linux targets, so no additional installation step is required at runtime either.

## Layout

```
qc_buffer_backend/
├── qc_buffer/               Core library (no accelerator SDK dependency)
│   ├── include/qc_buffer/
│   │   ├── rpcmem_loader.hpp           dlopen(libcdsprpc.so) + rpcmem_* symbols
│   │   ├── qc_buffer.hpp               RAII holder for one ION/dma-buf allocation
│   │   ├── qc_buffer_impl.hpp          rosidl::BufferImplBase<uint8_t> implementation
│   │   ├── qc_buffer_api.hpp           user API: allocate_buffer / get_data_ptr / get_dmabuf_fd
│   │   ├── fd_broker.hpp               per-publisher fd broker (SCM_RIGHTS delivery)
│   │   ├── intra_process_registry.hpp  uid -> shared_ptr<QcBuffer> table (retained for testing)
│   │   ├── qc_error.hpp                exception type
│   │   └── visibility_control.h
│   └── src/                            implementations of the above
├── qc_buffer_backend/                  pluginlib backend plugin (rosidl::BufferBackend)
│   ├── include/qc_buffer_backend/qc_buffer_backend.hpp
│   ├── src/qc_buffer_backend_plugin.cpp
│   └── qc_buffer_plugin.xml            plugin description registered with pluginlib
├── qc_buffer_backend_msgs/             QcBufferDescriptor.msg (published during serialization)
├── docs/                               design document
└── tutorial/                           minimal publisher/subscriber example (see tutorial/README.md)
```

## Using it in a publisher

Allocate a qc-backed buffer, write into it through the CPU pointer, and publish as usual. No handles or streams are needed — the memory is plain CPU memory.

```cpp
#include "qc_buffer/qc_buffer_api.hpp"

sensor_msgs::msg::Image msg;
msg.data = qc_buffer_backend::allocate_buffer(width * height * 3);
// ... set msg.width / height / encoding / step ...

uint8_t * p = qc_buffer_backend::get_data_ptr(msg.data);   // CPU pointer into ION memory
// ... fill p[0 .. size) ...

publisher_->publish(msg);
```

`allocate_buffer()` throws `qc_buffer_backend::QcError` if an explicit qc allocation fails. On a device without `libcdsprpc.so`, prefer writing through the normal `msg.data[i]` accessor so the code also works on the CPU fallback path (see the tutorial publisher for both branches).

## Using it in a subscriber

Accept any backend, then branch on `get_backend_type()`. When it is `"qc"` the payload is the same memory the publisher wrote (zero-copy); read it directly and/or hand its dma-buf fd to the HTP.

```cpp
#include "qc_buffer/qc_buffer_api.hpp"

rclcpp::SubscriptionOptions sub_opts;
sub_opts.acceptable_buffer_backends = "any";   // accept qc or cpu
subscription_ = create_subscription<sensor_msgs::msg::Image>(
  "qc_image", 10, callback, sub_opts);

// in the callback:
if (msg->data.get_backend_type() == "qc") {
  int fd = qc_buffer_backend::get_dmabuf_fd(msg->data);   // pass to the HTP accelerator
  uint8_t * p = qc_buffer_backend::get_data_ptr(msg->data);  // CPU read, no copy
  // ... use p / fd ...
} else {
  const std::vector<uint8_t> & data = msg->data;   // CPU fallback
  // ... use data ...
}
```

The backend is loaded by `pluginlib` at runtime; the subscriber only needs `acceptable_buffer_backends = "any"` (or `"qc"`) to receive qc-backed messages.

## Build

```bash
colcon build --packages-select qc_buffer_backend_msgs qc_buffer qc_buffer_backend
```

See `docs/qc_buffer_backend_design.md` for the design and `tutorial/README.md` for a runnable end-to-end example.
