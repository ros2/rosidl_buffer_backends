# onnxruntime_conversions

`onnxruntime_conversions` creates zero-copy ONNX Runtime tensor views over
`tensor_msgs/msg/ExperimentalTensor` storage.

## Usage

```cpp
auto memory_info =
  Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
auto msg = std::shared_ptr<onnxruntime_conversions::TensorMsg>(
  onnxruntime_conversions::allocate_tensor_msg(
    {1, 3, 224, 224}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

auto output =
  onnxruntime_conversions::from_output_tensor_msg(msg, memory_info);
io_binding.BindOutput("output", output.value());
```

`OrtTensorView` keeps the message storage alive. Keep the view alive while its
`Ort::Value` is bound or used by a session.

`ExperimentalTensor` carries DLPack-compatible dtype, shape, stride, and
offset metadata, but ONNX Runtime's public C++ API has no direct
`from_dlpack`. This package validates that metadata and uses
`Ort::Value::CreateTensor` to wrap the message's CPU pointer without copying.

This package is always CPU-only and uses the released `onnxruntime_vendor`
package from ros-controls. Use `onnxruntime_cuda_conversions` when both CPU and
CUDA-backed tensor messages must be handled by a CUDA-capable ONNX Runtime.

## Limitations

- Tensor layouts must be row-major contiguous.
- String and sub-byte element types are unsupported.
- Zero-copy output binding requires a known output shape.
- `to_tensor_msg` currently copies CPU tensors only.
