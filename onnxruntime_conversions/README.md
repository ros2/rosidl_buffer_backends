# ONNX Runtime conversions

Python and C++ libraries with a device-neutral API for converting between
`tensor_msgs/msg/ExperimentalTensor` messages and ONNX Runtime tensors
(`onnxruntime.OrtValue` in Python and `Ort::Value` in C++). CPU and accelerator
support is provided through separately installed plugins, so applications use
the same conversion API across devices. The API supports writing directly into
message storage, reading received messages as tensor views, and copying existing
tensors into messages. Accelerator plugins
use compatible `rosidl::Buffer` backends under the hood to share device memory,
enabling zero-copy transport between publishers and subscribers.

## Python

### Publisher

This example node demonstrates how to use `onnxruntime_conversions` in a
Python publisher. It runs an identity model once per second with a
`480 × 640 × 3` float32 input filled with `42`, writes inference output directly
into message storage, and publishes it on the `tensor` topic. The `model_path`
argument points to a model with input `input` and output `output`, both using
that shape and dtype:

```python
import numpy as np
import onnxruntime as ort
import onnxruntime_conversions
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor


class ExampleTensorPublisher(Node):

    def __init__(self, model_path):
        super().__init__('example_tensor_publisher')
        # Create a stream for the default plugin (empty on CPU).
        self.stream = onnxruntime_conversions.create_stream()
        # Configure inference to use the same backend and stream.
        self.session = ort.InferenceSession(
            model_path,
            providers=onnxruntime_conversions.session_providers(stream=self.stream))
        self.input_value = ort.OrtValue.ortvalue_from_numpy(
            np.full((480, 640, 3), 42.0, dtype=np.float32))
        self.publisher = self.create_publisher(ExperimentalTensor, 'tensor', 10)
        self.timer = self.create_timer(1.0, self.publish_tensor)

    def publish_tensor(self):
        # Allocate message storage on the stream's device.
        msg = onnxruntime_conversions.allocate_tensor_msg(
            (480, 640, 3), np.float32, stream=self.stream)
        # Create a writable view; output.value exposes its OrtValue.
        output = onnxruntime_conversions.from_output_tensor_msg(msg, self.stream)

        # Application code: run inference directly into message storage.
        binding = self.session.io_binding()
        binding.bind_ortvalue_input('input', self.input_value)
        binding.bind_ortvalue_output('output', output.value)
        self.session.run_with_iobinding(binding)

        self.publisher.publish(msg)
```

Replace the model path, input data, shape, dtype, and binding names with those
required by your model. Bind the model output to `output.value` so inference
writes into the message storage.

### Subscriber

This example node demonstrates how to use `onnxruntime_conversions` in a
Python subscriber. It reads each received tensor through a view and runs a
model that computes its mean, logging `42.0` for messages from the publisher
above. The `model_path` argument points to a model with a `480 × 640 × 3` float32
input named `input` and a single float32 mean value named `output`. The node
accepts both CPU and accelerator-backed buffers:

```python
import onnxruntime as ort
import onnxruntime_conversions
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor


class ExampleTensorSubscriber(Node):

    def __init__(self, model_path):
        super().__init__('example_tensor_subscriber')
        # Create a stream for the default plugin (empty on CPU).
        self.stream = onnxruntime_conversions.create_stream()
        # Configure inference to use the same backend and stream.
        self.session = ort.InferenceSession(
            model_path,
            providers=onnxruntime_conversions.session_providers(stream=self.stream))
        self.subscription = self.create_subscription(
            ExperimentalTensor,
            'tensor',
            self.receive_tensor,
            10,
            # Accept CPU and accelerator-backed message buffers.
            acceptable_buffer_backends='any',
        )

    def receive_tensor(self, received):
        # View the message storage directly; treat it as read-only.
        input_tensor = onnxruntime_conversions.from_input_tensor_msg(
            received, self.stream)

        # Application code: run the mean model and log its scalar output.
        binding = self.session.io_binding()
        binding.bind_ortvalue_input('input', input_tensor.value)
        binding.bind_output('output', 'cpu')
        self.session.run_with_iobinding(binding)
        mean = binding.copy_outputs_to_cpu()[0].item()

        self.get_logger().info(f'Mean pixel value: {mean:.1f}')
```

Replace the mean model, binding names, and logging with your own inference and
result processing.

The [standalone Python subscriber](onnxruntime_conversions_py/examples/ort_tensor_subscriber_node.py)
accepts the model path through its `model_path` ROS parameter.

### Existing tensors

This example copies an existing ONNX Runtime tensor into a new message and
publishes it:

```python
import onnxruntime_conversions

# Copy into a new message; the copy completes before this call returns.
# Use the tensor producer's stream, or synchronize with it first.
outgoing = onnxruntime_conversions.to_tensor_msg(ort_value, stream=stream)
publisher.publish(outgoing)
```

Replace `ort_value`, `stream`, and `publisher` with your application's tensor,
execution stream, and publisher.

### Install or build

Install the Debian packages:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py \
  ros-$ROS_DISTRO-onnxruntime-conversions-py-cpu
```

Or, from this repository's root after sourcing ROS, build from source:

```bash
rosdep install --from-paths \
  $(colcon list --packages-up-to onnxruntime_conversions_py_cpu --paths-only) \
  --ignore-src -y
colcon build --merge-install --packages-up-to onnxruntime_conversions_py_cpu
source install/setup.bash
```

## C++

### Publisher

This example node demonstrates how to use `onnxruntime_conversions` in a
C++ publisher. It runs an identity model once per second with a
`480 × 640 × 3` float32 input filled with `42`, writes inference output directly
into message storage, and publishes it on the `tensor` topic. The `model_path`
argument points to a model with input `input` and output `output`, both using
that shape and dtype:

```cpp
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"
#include "rclcpp/rclcpp.hpp"

class ExampleTensorPublisher : public rclcpp::Node
{
public:
  explicit ExampleTensorPublisher(const std::string & model_path)
  : Node("example_tensor_publisher"),
    env_(ORT_LOGGING_LEVEL_WARNING, "example_tensor_publisher"),
    // Create a stream for the default plugin (empty on CPU).
    stream_(onnxruntime_conversions::create_stream(env_)),
    input_data_(480 * 640 * 3, 42.0f)
  {
    Ort::SessionOptions options;
    // Configure inference to use the same backend and stream.
    onnxruntime_conversions::configure_session_options(options, stream_);
    session_ = Ort::Session(env_, model_path.c_str(), options);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_value_ = Ort::Value::CreateTensor<float>(
      memory, input_data_.data(), input_data_.size(), shape_.data(), shape_.size());
    publisher_ = create_publisher<onnxruntime_conversions::TensorMsg>("tensor", 10);
    timer_ = create_wall_timer(
      std::chrono::seconds(1), [this]() {publish_tensor();});
  }

private:
  void publish_tensor()
  {
    // Allocate message storage on the stream's device.
    auto msg = onnxruntime_conversions::allocate_tensor_msg(
      shape_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, stream_);
    {
      // Create a writable view; output.value() exposes its Ort::Value.
      auto output = onnxruntime_conversions::from_output_tensor_msg(*msg, stream_);

      // Application code: run inference directly into message storage.
      Ort::IoBinding binding(session_);
      binding.BindInput("input", input_value_);
      binding.BindOutput("output", output.value());
      session_.Run(Ort::RunOptions{}, binding);
    }  // Release the binding and view before handing the message to the publisher.

    publisher_->publish(std::move(msg));
  }

  Ort::Env env_;
  onnxruntime_conversions::Stream stream_;
  Ort::Session session_{nullptr};
  const std::vector<int64_t> shape_{480, 640, 3};
  std::vector<float> input_data_;
  Ort::Value input_value_{nullptr};
  rclcpp::Publisher<onnxruntime_conversions::TensorMsg>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};
```

Replace the model path, input data, shape, dtype, and binding names with those
required by your model. Bind the model output to `output.value()` so inference
writes into the message storage.

### Subscriber

This example node demonstrates how to use `onnxruntime_conversions` in a
C++ subscriber. It reads each received tensor through a view and runs a
model that computes its mean, logging `42.0` for messages from the publisher
above. The `model_path` argument points to a model with a `480 × 640 × 3` float32
input named `input` and a single float32 mean value named `output`. The node
accepts both CPU and accelerator-backed buffers:

```cpp
#include <string>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"
#include "rclcpp/rclcpp.hpp"

class ExampleTensorSubscriber : public rclcpp::Node
{
public:
  explicit ExampleTensorSubscriber(const std::string & model_path)
  : Node("example_tensor_subscriber"),
    env_(ORT_LOGGING_LEVEL_WARNING, "example_tensor_subscriber"),
    // Create a stream for the default plugin (empty on CPU).
    stream_(onnxruntime_conversions::create_stream(env_))
  {
    Ort::SessionOptions session_options;
    // Configure inference to use the same backend and stream.
    onnxruntime_conversions::configure_session_options(session_options, stream_);
    session_ = Ort::Session(env_, model_path.c_str(), session_options);
    rclcpp::SubscriptionOptions options;
    // Accept CPU and accelerator-backed message buffers.
    options.acceptable_buffer_backends = "any";
    subscription_ = create_subscription<onnxruntime_conversions::TensorMsg>(
      "tensor", 10,
      [this](onnxruntime_conversions::TensorMsg::ConstSharedPtr received) {
        receive_tensor(*received);
      }, options);
  }

private:
  void receive_tensor(const onnxruntime_conversions::TensorMsg & received)
  {
    // View the message storage directly; treat it as read-only.
    auto input_tensor = onnxruntime_conversions::from_input_tensor_msg(received, stream_);

    // Application code: run the mean model and log its scalar output.
    Ort::IoBinding binding(session_);
    binding.BindInput("input", input_tensor.value());
    binding.BindOutput(
      "output", Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    session_.Run(Ort::RunOptions{}, binding);
    const auto outputs = binding.GetOutputValues();
    const auto mean = outputs.front().GetTensorData<float>()[0];

    RCLCPP_INFO(get_logger(), "Mean pixel value: %.1f", static_cast<double>(mean));
  }

  Ort::Env env_;
  onnxruntime_conversions::Stream stream_;
  Ort::Session session_{nullptr};
  rclcpp::Subscription<onnxruntime_conversions::TensorMsg>::SharedPtr subscription_;
};
```

Replace the mean model, binding names, and logging with your own inference and
result processing.

### Existing tensors

This example copies an existing ONNX Runtime tensor into a new message and
publishes it:

```cpp
// Copy into a new message; the copy completes before this call returns.
// Use the tensor producer's stream, or synchronize with it first.
auto outgoing = onnxruntime_conversions::to_tensor_msg(ort_value, stream);
publisher->publish(std::move(outgoing));
```

Replace `ort_value`, `stream`, and `publisher` with your application's tensor,
execution stream, and publisher.

### Install or build

Install the Debian packages:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions \
  ros-$ROS_DISTRO-onnxruntime-conversions-cpu
```

Or, from this repository's root after sourcing ROS, build from source:

```bash
rosdep install --from-paths \
  $(colcon list --packages-up-to onnxruntime_conversions_cpu --paths-only) \
  --ignore-src -y
colcon build --merge-install --packages-up-to onnxruntime_conversions_cpu
source install/setup.bash
```

## Accelerated devices

With the CPU packages above installed, the examples run on CPU by default when
no accelerator plugin is installed. To use an accelerated device, install or
build the corresponding plugin for the language you use. Plugins can be added
without rebuilding the conversion core or your application.

Accelerator backends enable **zero-copy transport from publisher to subscriber**
when they can share device memory: the subscriber accesses the publisher's tensor
payload without copying it or transferring it to host memory. The publisher
example binds inference output to message storage, and the subscriber binds
that storage directly as inference input.

For example, install the CUDA plugin from Debian packages:

```bash
# Python
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cuda
# C++
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-cuda
```

Or, from this repository's root, build the plugin for your language. Each
rosdep command selects only that plugin and its dependencies:

Python:

```bash
rosdep install --from-paths \
  $(colcon list --packages-up-to onnxruntime_conversions_py_cuda --paths-only) \
  --ignore-src -y
colcon build --merge-install --packages-up-to onnxruntime_conversions_py_cuda
source install/setup.bash
```

C++:

```bash
rosdep install --from-paths \
  $(colcon list --packages-up-to onnxruntime_conversions_cuda --paths-only) \
  --ignore-src -y
colcon build --merge-install --packages-up-to onnxruntime_conversions_cuda
source install/setup.bash
```

C++ discovers plugins on first registry use; Python discovers them when the
module is imported. After installing or building a plugin, source the ROS
environment and restart the application.

When the device and a compatible ONNX Runtime provider are available, the plugin
participates in automatic backend selection. Each plugin defines its priority in
its implementation. Currently, the CPU plugin has priority `0` and the CUDA
example plugin has priority `100`, in both Python and C++. Stream creation,
allocation, and session setup use the available plugin with the highest priority.
Pass an explicit `backend` to `create_stream` to select the backend for a node.

Use the same stream for conversions and inference. Applications with an existing
native stream can wrap it with `borrow_stream`; keep its owner alive while the
session and tensor views use it. In C++, the `Ort::Env` must outlive the stream.

## Requirements and package structure

### Requirements

| Component | Debian packages | Source builds |
| --- | --- | --- |
| Ubuntu / ROS | Ubuntu **26.04** (Resolute), with ROS 2 Lyrical or Rolling. | A ROS 2 Lyrical or Rolling environment. On Ubuntu **24.04** (Noble), build ROS 2 Lyrical from source. |
| ONNX Runtime | Version **1.29.0**, supplied by the ROS vendor packages. | Reuses compatible installations of **>=1.23.0 for C++** or **>=1.26.0 for Python**; otherwise, the vendor packages download **1.29.0**. |
| CUDA Toolkit (example plugin) | The `cuda-toolkit` dependency, installed through APT. | An installed toolkit compatible with the selected provider (**>=13.1** for the pinned fallback); `rosdep` can install `cuda-toolkit` if needed. |
| cuDNN (example plugin) | Version **9.24.0.43**, supplied by the CUDA vendor package. | A reused ONNX Runtime provider needs its matching cuDNN libraries. The pinned fallback reuses compatible cuDNN **>=9.24,<10** for the toolkit's CUDA major version, or supplies **9.24.0.43**. |

The CUDA plugin requires a CUDA-enabled ONNX Runtime provider. CPU plugins do
not require CUDA or cuDNN. Reusing a C++ SDK requires its headers, libraries, and
version metadata (`onnxruntimeConfigVersion.cmake` or `VERSION_NUMBER`).

Both APIs require contiguous, row-major tensors. Explicit strides must match
the contiguous strides for the shape; an empty `strides` field uses that layout.
A nonzero `byte_offset` is supported when the tensor fits within the buffer.

For Ubuntu 24.04 source builds, see the
[ROS 2 Lyrical platform requirements](https://github.com/ros2/ros2_documentation/blob/lyrical/source/Releases/lyrical/supported-platforms.rst).

### Package structure

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `onnxruntime_conversions` | `onnxruntime_conversions_py` |
| CPU plugin | `onnxruntime_conversions_cpu` | `onnxruntime_conversions_py_cpu` |
| CUDA plugin | `onnxruntime_conversions_cuda` | `onnxruntime_conversions_py_cuda` |

The core packages install example nodes from their `examples/` directories.
Backend-dependent unit and publication tests live in the corresponding CPU and
CUDA plugin packages.

## License

Apache-2.0
