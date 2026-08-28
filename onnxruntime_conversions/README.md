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

`OrtTensorView` keeps the message storage and any CUDA buffer handle alive.
Keep the view alive while its `Ort::Value` is bound or used by a session.

CPU buffers are always supported. If `cuda_buffer` is available while this
package is built, CUDA buffers are also supported. Pass the same CUDA stream
to the conversion functions that is configured as ONNX Runtime's
`user_compute_stream`.

## Limitations

- Tensor layouts must be row-major contiguous.
- String and sub-byte element types are unsupported.
- Zero-copy output binding requires a known output shape.
- `to_tensor_msg` currently copies CPU tensors only.
