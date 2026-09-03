# onnxruntime_conversions

`onnxruntime_conversions` provides the compiled C++ API, plugin registry, and
required CPU backend plugin for zero-copy ONNX Runtime tensor views over
`tensor_msgs/msg/ExperimentalTensor` storage.

```cmake
find_package(onnxruntime_conversions REQUIRED)
target_link_libraries(my_target
  onnxruntime_conversions::onnxruntime_conversions)
```

```cpp
auto memory_info =
  Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
auto msg = std::shared_ptr<onnxruntime_conversions::TensorMsg>(
  onnxruntime_conversions::allocate_tensor_msg(
    {1, 3, 224, 224}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu"));
auto output =
  onnxruntime_conversions::from_output_tensor_msg(msg, memory_info);
io_binding.BindOutput("output", output.value());
```

Backends are discovered through pluginlib. The CPU backend is a shared plugin
bundled in this ROS package and is mandatory. The optional
`onnxruntime_conversions_cuda_plugin` package registers the `cuda` backend.

`auto` selects an accelerator only when its capability probe accepts the
configuration. Otherwise it selects CPU. Explicit `cpu` and `cuda` selection
are strict and never fall back. Ambiguous automatic selection reports an
error. Plugin libraries are pinned for the process lifetime so plugin-defined
leases remain safe during static teardown.

For CUDA, install `onnxruntime_core_vendor`, `onnxruntime_cuda_vendor`,
`onnxruntime_conversions`, and `onnxruntime_conversions_cuda_plugin`. The two
native ONNX Runtime vendors must share a merged install prefix.

```bash
colcon build --merge-install --packages-up-to onnxruntime_conversions
colcon build --merge-install --packages-up-to onnxruntime_conversions_cuda_plugin
```

## Limitations

- Tensor layouts must be row-major contiguous.
- String and sub-byte element types are unsupported.
- Zero-copy output binding requires a known output shape.
- CUDA operations require the optional plugin and a non-null explicit stream.
