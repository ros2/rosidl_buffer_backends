# PyTorch conversions

Python and C++ libraries with a device-neutral API for converting between
`tensor_msgs/msg/ExperimentalTensor` messages and PyTorch tensors. CPU and
accelerator support is provided through separately installed plugins, so
applications use the same conversion API across devices.
The API supports writing directly into message storage, reading received messages
as tensor views, and copying existing tensors into messages. Accelerator plugins
use compatible `rosidl::Buffer` backends under the hood to share device memory,
enabling zero-copy transport between publishers and subscribers.

## Python

### Publisher

This example node demonstrates how to use `torch_conversions` in a Python
publisher. It publishes a `480 × 640 × 3` tensor of unsigned 8-bit values on the
`tensor` topic once per second. Each timer callback allocates a message, creates
a writable tensor view of its storage, fills every element with `42`, and
publishes the message:

```python
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor
import torch
import torch_conversions


class ExampleTensorPublisher(Node):

    def __init__(self):
        super().__init__('example_tensor_publisher')
        self.publisher = self.create_publisher(ExperimentalTensor, 'tensor', 10)
        self.timer = self.create_timer(1.0, self.publish_tensor)

    def publish_tensor(self):
        # Select a stream for this callback (a no-op on CPU).
        with torch_conversions.set_stream():
            # Allocate a message with tensor storage on the default device.
            msg = torch_conversions.allocate_tensor_msg((480, 640, 3), torch.uint8)

            # Create a writable tensor view of msg.data.
            output = torch_conversions.from_output_tensor_msg(msg)

            # Application code: fill the message-backed tensor in place.
            output.fill_(42)

            self.publisher.publish(msg)
```

Replace `output.fill_(42)` with your own code that writes into `output`. If it
produces a separate tensor, use `output.copy_(result)` to fill the message storage.

### Subscriber

This example node demonstrates how to use `torch_conversions` in a Python
subscriber. It subscribes to the `tensor` topic and reads each received tensor
through a tensor view to calculate and log its mean value. It logs `42.0` for
messages from the publisher above and accepts both CPU and accelerator-backed
buffers:

```python
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor
import torch_conversions


class ExampleTensorSubscriber(Node):

    def __init__(self):
        super().__init__('example_tensor_subscriber')
        self.subscription = self.create_subscription(
            ExperimentalTensor,
            'tensor',
            self.receive_tensor,
            10,
            # Accept CPU and accelerator-backed message buffers.
            acceptable_buffer_backends='any',
        )

    def receive_tensor(self, received):
        # Select a stream for this callback (a no-op on CPU).
        with torch_conversions.set_stream():
            # View the message storage directly; treat it as read-only.
            # The default clone=True returns an independent tensor instead.
            input_tensor = torch_conversions.from_input_tensor_msg(
                received, clone=False)

            # Application code: read the shared tensor without modifying it.
            mean = input_tensor.float().mean().item()

            self.get_logger().info(f'Mean pixel value: {mean:.1f}')
```

Replace the mean calculation and logging with your own processing of
`input_tensor`.

### Existing tensors

This example copies an existing tensor into a new message and publishes it:

```python
import torch_conversions

# Copy into a new message; the copy completes before this call returns.
# Use the tensor producer's stream, or synchronize with it first.
outgoing = torch_conversions.to_tensor_msg(tensor)
publisher.publish(outgoing)
```

Replace `tensor` with your application's tensor and `publisher` with its
publisher.

### Install or build

Install the Debian packages:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-py \
  ros-$ROS_DISTRO-torch-conversions-py-cpu
```

Or, from this repository's root after sourcing ROS, build from source:

```bash
rosdep install --from-paths \
  tensor_msgs \
  torch_vendor/python3_torch_vendor \
  torch_conversions/torch_conversions_py \
  torch_conversions/torch_conversions_py_cpu \
  --ignore-src -y
colcon build --merge-install --packages-up-to torch_conversions_py_cpu
source install/setup.bash
```

## C++

### Publisher

This example node demonstrates how to use `torch_conversions` in a C++
publisher. It publishes a `480 × 640 × 3` tensor of unsigned 8-bit values on the
`tensor` topic once per second. Each timer callback allocates a message, creates
a writable tensor view of its storage, fills every element with `42`, and
publishes the message:

```cpp
#include <chrono>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"
#include "torch_conversions/torch_conversions.hpp"

class ExampleTensorPublisher : public rclcpp::Node
{
public:
  ExampleTensorPublisher()
  : Node("example_tensor_publisher")
  {
    publisher_ = create_publisher<tensor_msgs::msg::ExperimentalTensor>("tensor", 10);
    timer_ = create_wall_timer(
      std::chrono::seconds(1), [this]() {publish_tensor();});
  }

private:
  void publish_tensor()
  {
    // Select a stream for this callback (a no-op on CPU).
    auto guard = torch_conversions::set_stream();

    // Allocate a message with tensor storage on the default device.
    auto msg = torch_conversions::allocate_tensor_msg({480, 640, 3}, torch::kUInt8);
    {
      // Create a writable tensor view of msg->data.
      auto output = torch_conversions::from_output_tensor_msg(*msg);

      // Application code: fill the message-backed tensor in place.
      output.fill_(42);
    }  // Release the view before handing the message to the publisher.

    publisher_->publish(std::move(msg));
  }

  rclcpp::Publisher<tensor_msgs::msg::ExperimentalTensor>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};
```

Replace `output.fill_(42)` with your own code that writes into `output`. If it
produces a separate tensor, use `output.copy_(result)` to fill the message storage.

### Subscriber

This example node demonstrates how to use `torch_conversions` in a C++
subscriber. It subscribes to the `tensor` topic and reads each received tensor
through a tensor view to calculate and log its mean value. It logs `42.0` for
messages from the publisher above and accepts both CPU and accelerator-backed
buffers:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"
#include "torch_conversions/torch_conversions.hpp"

class ExampleTensorSubscriber : public rclcpp::Node
{
public:
  ExampleTensorSubscriber()
  : Node("example_tensor_subscriber")
  {
    rclcpp::SubscriptionOptions options;
    // Accept CPU and accelerator-backed message buffers.
    options.acceptable_buffer_backends = "any";
    subscription_ = create_subscription<tensor_msgs::msg::ExperimentalTensor>(
      "tensor", 10,
      [this](tensor_msgs::msg::ExperimentalTensor::ConstSharedPtr received) {
        receive_tensor(*received);
      }, options);
  }

private:
  void receive_tensor(const tensor_msgs::msg::ExperimentalTensor & received)
  {
    // Select a stream for this callback (a no-op on CPU).
    auto guard = torch_conversions::set_stream();

    // View the message storage directly; treat it as read-only.
    // The default clone=true returns an independent tensor instead.
    auto input_tensor = torch_conversions::from_input_tensor_msg(received, /*clone=*/false);

    // Application code: read the shared tensor without modifying it.
    const auto mean = input_tensor.to(torch::kFloat32).mean().item<double>();

    RCLCPP_INFO(get_logger(), "Mean pixel value: %.1f", mean);
  }

  rclcpp::Subscription<tensor_msgs::msg::ExperimentalTensor>::SharedPtr subscription_;
};
```

Replace the mean calculation and logging with your own processing of
`input_tensor`.

### Existing tensors

This example copies an existing tensor into a new message and publishes it:

```cpp
// Copy into a new message; the copy completes before this call returns.
// Use the tensor producer's stream, or synchronize with it first.
auto outgoing = torch_conversions::to_tensor_msg(tensor);
publisher->publish(std::move(outgoing));
```

Replace `tensor` with your application's tensor and `publisher` with its
publisher.

### Install or build

Install the Debian packages:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions \
  ros-$ROS_DISTRO-torch-conversions-cpu
```

Or, from this repository's root after sourcing ROS, build from source:

```bash
rosdep install --from-paths \
  tensor_msgs \
  torch_vendor/libtorch_vendor \
  torch_conversions/torch_conversions \
  torch_conversions/torch_conversions_cpu \
  --ignore-src -y
colcon build --merge-install --packages-up-to torch_conversions_cpu
source install/setup.bash
```

## Accelerated devices

With the CPU packages above installed, the examples run on CPU by default when
no accelerator plugin is installed. To use an accelerated device, install or
build the corresponding plugin for the language you use. Plugins can be added
without rebuilding the conversion core or your application.

Accelerator backends enable **zero-copy transport from publisher to subscriber**
when they can share device memory: the subscriber accesses the publisher's tensor
payload without copying it or transferring it to host memory. The examples write
directly into that message storage and read it through a tensor view with cloning
disabled.

For example, install the CUDA plugin from Debian packages:

```bash
# Python
sudo apt install ros-$ROS_DISTRO-torch-conversions-py-cuda
# C++
sudo apt install ros-$ROS_DISTRO-torch-conversions-cuda
```

Or, from this repository's root, build the plugin for your language. Each
rosdep command selects only that plugin and its dependencies:

Python:

```bash
rosdep install --from-paths \
  $(colcon list --packages-up-to torch_conversions_py_cuda --paths-only) \
  --ignore-src -y
colcon build --merge-install --packages-up-to torch_conversions_py_cuda
source install/setup.bash
```

C++:

```bash
rosdep install --from-paths \
  $(colcon list --packages-up-to torch_conversions_cuda --paths-only) \
  --ignore-src -y
colcon build --merge-install --packages-up-to torch_conversions_cuda
source install/setup.bash
```

C++ discovers plugins on first registry use; Python discovers them when the
module is imported. After installing or building a plugin, source the ROS
environment and restart the application.

When the device and a compatible Torch provider are available, the plugin
participates in automatic device selection. Each plugin defines its priority in
its implementation. Currently, the CPU plugin has priority `0` and the CUDA
example plugin has priority `100`, in both Python and C++. New allocations use
the available plugin with the highest priority, so the example accelerator is
preferred over CPU when available.

Pass an explicit device to `allocate_tensor_msg` to override automatic selection
for that allocation.

## Requirements and package structure

### Requirements

| Component | Debian packages | Source builds |
| --- | --- | --- |
| Ubuntu / ROS | Ubuntu **26.04** (Resolute), with ROS 2 Lyrical or Rolling. | A ROS 2 Lyrical or Rolling environment. On Ubuntu **24.04** (Noble), build ROS 2 Lyrical from source. |
| PyTorch / LibTorch | Version **2.14.0**, supplied by the ROS vendor packages. | Reuses a compatible installation of version **2.5.0 or newer**; otherwise, the vendor packages download **2.14.0**. |
| CUDA Toolkit (example plugin) | The `cuda-toolkit` dependency, installed through APT. | An installed toolkit compatible with the selected provider (**>=13.1** for the pinned fallback); `rosdep` can install `cuda-toolkit` if needed. |

The CUDA plugin requires a CUDA-enabled PyTorch / LibTorch build, supplied by
the CUDA vendor packages or reused during a source build. CPU plugins do not
require CUDA.

For Ubuntu 24.04 source builds, see the
[ROS 2 Lyrical platform requirements](https://github.com/ros2/ros2_documentation/blob/lyrical/source/Releases/lyrical/supported-platforms.rst).

### Package structure

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `torch_conversions` | `torch_conversions_py` |
| CPU plugin | `torch_conversions_cpu` | `torch_conversions_py_cpu` |
| CUDA plugin | `torch_conversions_cuda` | `torch_conversions_py_cuda` |

## License

Apache-2.0
