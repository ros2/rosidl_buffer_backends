# qc_buffer_backend tutorial

A minimal pipeline showing `qc_buffer_backend` HTP–CPU zero-copy: one publisher node and one subscriber node.

- `src/qc_image_publisher.cpp` — allocates a qc-backed buffer, writes it via the CPU pointer, and publishes it on the `qc_image` topic.
- `src/qc_image_subscriber.cpp` — subscribes to `qc_image`, reads the same memory directly (zero-copy) and prints its dma-buf fd.
- `launch/qc_buffer_tutorial.launch.py` — loads both nodes into a single `component_container` process.

## Build

```bash
colcon build --packages-up-to qc_buffer_tutorial
```

## Run

```bash
source install/setup.bash
ros2 launch qc_buffer_tutorial qc_buffer_tutorial.launch.py
```

Expected output (publisher every 10 frames, subscriber every frame):

```
[qc_image_publisher]  published 10 images (backend: qc, fd: 42)
[qc_image_subscriber] image #1: 8x8, 192 bytes, backend=qc, dmabuf_fd=43, first_byte=0 ...
```

`backend=qc` on the subscriber side confirms the payload was delivered through dma-buf without a copy. The reported `dmabuf_fd` is what an application passes to the HTP accelerator to let it read the same memory. The subscriber receives a fresh fd via SCM_RIGHTS and mmap()s it to the same physical pages the publisher wrote.

If you see `backend=cpu (CPU fallback)` instead, the device does not have `libcdsprpc.so` available.
