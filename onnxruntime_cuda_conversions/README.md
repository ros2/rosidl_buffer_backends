# onnxruntime_cuda_conversions

`onnxruntime_cuda_conversions` creates zero-copy ONNX Runtime tensor views over
CPU or CUDA-backed `tensor_msgs/msg/ExperimentalTensor` storage. It always uses
the CUDA 13 ONNX Runtime supplied by `onnxruntime_cuda_vendor`; CUDA support is
not optional or auto-detected.

```cpp
Ort::MemoryInfo memory_info(
  "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
auto msg = std::shared_ptr<onnxruntime_cuda_conversions::TensorMsg>(
  onnxruntime_cuda_conversions::allocate_tensor_msg(
    {1, 3, 224, 224}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"));

auto output = onnxruntime_cuda_conversions::from_output_tensor_msg(
  msg, memory_info, cuda_stream);
io_binding.BindOutput("output", output.value());
```

`OrtTensorView` keeps the message storage and CUDA synchronization handle
alive. Keep the view alive while its `Ort::Value` is bound or used by a
session. Use the same CUDA stream for conversions and ONNX Runtime's
`user_compute_stream`.

`ExperimentalTensor` carries DLPack-compatible dtype, shape, stride, and
offset metadata, but ONNX Runtime's public C++ API has no direct
`from_dlpack`. This package validates that metadata and uses
`Ort::Value::CreateTensor` to wrap the message's CPU or CUDA pointer without
copying.

## Limitations

- Tensor layouts must be row-major contiguous.
- String and sub-byte element types are unsupported.
- Zero-copy output binding requires a known output shape.
- `to_tensor_msg` currently copies CPU tensors only.
