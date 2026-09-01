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
`Ort::Value::CreateTensor` to wrap the message's pointer without copying.

CPU conversion support is always available. CUDA-buffer support is compiled
only when `onnxruntime_gpu_vendor` selected a CUDA archive and both
`CUDAToolkit` and `cuda_buffer` are available. Pass `"cuda"` to
`allocate_tensor_msg` and the execution stream to the view functions when
using CUDA storage.

## Limitations

- Tensor layouts must be row-major contiguous.
- String and sub-byte element types are unsupported.
- Zero-copy output binding requires a known output shape.
- `to_tensor_msg` currently copies CPU tensors only.
