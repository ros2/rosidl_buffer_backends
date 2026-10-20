# torch_conversions (DLPack-aligned)

Header-only helper library that converts between DLPack-shaped ROS 2
messages (`tensor_msgs/ExperimentalTensor` and native `sensor_msgs/Image`) and an `at::Tensor`, riding on top of
whichever `rosidl::Buffer` storage backend is registered at runtime.

The message schema follows [DLPack](https://dmlc.github.io/dlpack/latest/)
tensor metadata, so any DLPack-compatible framework (PyTorch, TensorFlow,
JAX, CuPy, ONNX Runtime, MXNet, RAPIDS, ...) can plug in via a thin wrapper
and interoperate over the wire without re-encoding shape / dtype metadata.

> **Status: experimental.** The message is named `ExperimentalTensor` on
> purpose. The schema is used internally to validate the buffer-backend
> design and may change before it is renamed to `Tensor` and stabilized.

## Packages

| Package | Description |
|---|---|
| `tensor_msgs` | `ExperimentalTensor.msg` definition: DLPack-aligned `{dtype_code, dtype_bits, dtype_lanes}`, `shape[]`, `strides[]`, `byte_offset`, `data[]`. |
| `torch_conversions` | Header-only library: allocation, `at::Tensor` ↔ `ExperimentalTensor.msg` or native `sensor_msgs/Image` conversion, DLPack export, and CUDA stream helpers. |

The `uint8[] data` field maps to `rosidl::Buffer<uint8_t>`,
so storage and transport are delegated to whichever buffer backend is
registered for the connection.

## The `ExperimentalTensor.msg` schema

```
# DLDataType
uint8  dtype_code        # DLPack DLDataTypeCode: 0=Int, 1=UInt, 2=Float, 4=BFloat, 6=Bool, ...
uint8  dtype_bits        # 8, 16, 32, 64, ...
uint16 dtype_lanes       # SIMD lanes; 1 for plain scalar

# DLTensor
int64[] shape
int64[] strides          # empty = contiguous (DLPack nullptr convention)
uint64  byte_offset      # view offset into `data`

# Underlying storage (may be larger than numel * element_size for views)
uint8[] data
```

The message carries DLPack's dtype / shape / stride / offset metadata. The
`DLDevice` fields are derived from the underlying `msg.data` buffer backend.

## Build

```bash
# CUDA path (recommended): build cuda_buffer_backend first.
colcon build --symlink-install --packages-up-to cuda_buffer_backend
source install/setup.sh

colcon build --symlink-install --packages-up-to torch_conversions
source install/setup.sh
```

## Testing

```bash
colcon test --packages-select tensor_msgs torch_conversions
colcon test-result --verbose
```

## Examples

### Publisher

```cpp
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

void timer_cb()
{
  // Uses a non-default CUDA stream when CUDA is available; no-op on CPU.
  auto guard = torch_conversions::set_stream();

  // Pre-sizes msg.data and fills DLPack shape / dtype metadata.
  // Uses the accelerated buffer backend when available, otherwise CPU.
  auto msg = torch_conversions::allocate_tensor_msg(
    {height, width, 3}, torch::kByte);

  {
    // Output path: writable tensor view that aliases msg.data.
    at::Tensor t = torch_conversions::from_output_tensor_msg(*msg);
    render_pipeline(t);
  }

  publisher_->publish(std::move(msg));
}
```

### Subscriber

```cpp
void cb(const tensor_msgs::msg::ExperimentalTensor::SharedPtr msg)
{
  // Uses the same stream discipline as the publisher side.
  auto guard = torch_conversions::set_stream();

  // Default clone=true: independent tensor, safe to mutate.
  at::Tensor in = torch_conversions::from_input_tensor_msg(*msg);
  auto out = model_(in);

  // Or clone=false for a zero-copy read-only view.
  at::Tensor view = torch_conversions::from_input_tensor_msg(*msg, /*clone=*/false);
}
```

### Publishing an existing tensor

```cpp
at::Tensor t = compute_something().contiguous();

// Allocates a Tensor message, copies tensor data, and fills metadata.
auto msg = torch_conversions::to_tensor_msg(t);
publisher_->publish(std::move(msg));
```

`to_tensor_msg(t)` allocates a message, copies tensor data into `msg.data`,
and updates shape / strides / dtype metadata to match `t`. Use
`to_tensor_msg(*msg, t)` when you want to reuse a pre-sized message buffer.

## Native `sensor_msgs/Image`

The same buffer-backed conversion path supports packed, byte-oriented native
image messages. `Image.data` is a `rosidl::Buffer<uint8_t>`, so requesting CUDA
storage places the pixels in the CUDA buffer backend while preserving the
standard ROS image metadata and wire type.

```cpp
#include "sensor_msgs/image_encodings.hpp"
#include "torch_conversions/torch_conversions.hpp"

auto guard = torch_conversions::set_stream();

// Initializes height, width, encoding, endianness, step, and an exact-size
// data allocation. CUDA is selected explicitly here.
auto msg = torch_conversions::allocate_image_msg(
  height, width, sensor_msgs::image_encodings::RGB8, c10::kCUDA);

// Copy an existing HWC uint8 tensor into Image.data. This is stream-ordered;
// the conversion does not synchronize the CUDA device or stream.
torch_conversions::to_image_msg(*msg, source_hwc);
publisher_->publish(std::move(msg));
```

A subscriber obtains a tensor from the standard Image fields:

```cpp
void cb(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  auto guard = torch_conversions::set_stream();

  // HWC shape and row stride are derived from height, width, encoding, and
  // step. false requests a zero-copy read-only view.
  at::Tensor input =
    torch_conversions::from_input_image_msg(*msg, /*clone=*/false);
  at::Tensor output = resize(input);
}
```

The Image API consists of:

- `allocate_image_msg(height, width, encoding, device)` for initialized CPU or
  CUDA-backed storage.
- `from_output_image_msg(msg)` for a writable HWC tensor view used by producers.
- `from_input_image_msg(msg, clone)` for subscriber input. With `clone=false`,
  the CUDA read handle and event dependency remain attached to the tensor
  lifetime and no pixel copy is made.
- `to_image_msg(msg, tensor)` to copy into an existing allocation without
  changing its header or image metadata, and `to_image_msg(tensor, encoding)`
  to allocate and copy in one call.

Conversions do not call `cudaDeviceSynchronize` or `cudaStreamSynchronize`.
CUDA ReadHandle and WriteHandle objects keep storage and stream-event ordering
alive until the corresponding tensor view is destroyed. The current bridge
accepts packed byte encodings such as `mono8`, `rgb8`, `bgr8`, `rgba8`, Bayer8,
YUV422, and `8UC`/`8SC`. It rejects planar formats, non-byte encodings, a step
smaller than the packed row size, or data shorter than `height * step` with a
descriptive exception. Row padding is preserved through tensor strides.

## License

Apache-2.0
