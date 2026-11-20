# PyTorch conversions

Python and C++ libraries with a device-neutral API for converting between
`tensor_msgs/msg/ExperimentalTensor` messages and PyTorch tensors. CPU and
accelerator support is provided through separately installed plugins, so
applications use the same conversion API across devices.
The API supports writing directly into message storage, reading received messages
as tensor views, and copying existing tensors into messages. Accelerator plugins
use compatible `rosidl::Buffer` backends under the hood to share device memory,
enabling zero-copy transport between publishers and subscribers.

## Plugin architecture

Applications call the public C++ `torch_conversions` API or the Python
`torch_conversions` module (installed by the ROS package `torch_conversions_py`).
The cores handle tensor metadata and dispatch operations to separately packaged
conversion plugins. Applications do not import or link a specific plugin.

| Layer | Responsibility | CUDA example |
| --- | --- | --- |
| Conversion core | Public API, metadata validation, and runtime plugin discovery | `torch_conversions` / `torch_conversions_py` |
| Torch conversion plugin | Allocate storage, create Torch views, copy tensors, and select execution streams | `torch_conversions_cuda` / `torch_conversions_py_cuda` |
| Buffer backend and access API | Own or map memory, coordinate access, and support ROS transport | `cuda_buffer`, `cuda_buffer_backend`, and `cuda_buffer_py` |
| Torch provider | Supply a compatible Torch build with the required device support | `libtorch_cuda_vendor` / `python3_torch_cuda_vendor` |

A conversion plugin adapts an existing buffer backend to Torch; it does not by
itself implement ROS transport or add a new device type to Torch. A vendor needs
both a compatible Torch implementation and buffer storage/access support.
The C++ and Python registries are independent: install a plugin for each language
used by your application.

C++ discovers shared libraries through `pluginlib`. Python discovers registration
functions through the ament resource index. Installing a plugin and sourcing its
workspace makes it discoverable; no application-side registration is needed.
Install the CPU plugin for host storage support, and add an accelerator plugin
and its dependencies for device storage. The core alone provides no storage
implementation. See [Accelerated devices](#accelerated-devices) for installation
and discovery timing, and [Developing a plugin](#developing-a-plugin) for vendor
integration.

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
            del output  # Release the writable view before publication.

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

A runnable Python subscriber is available in
[`examples/torch_tensor_subscriber_node.py`](torch_conversions_py/examples/torch_tensor_subscriber_node.py).

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

### Inspect and select a device

Use `available_backends()` and `backend_available("cuda")` to inspect the usable
plugins in this process. Installed plugins whose runtime or device is unavailable
may be absent. `default_backend()` reports the automatic selection and raises an
error if no usable plugin exists. `backend_for_device` maps a Torch device type to
its registered storage backend.

Pass an explicit device to both `set_stream` and `allocate_tensor_msg` when you
want to override the default. For example, force CPU even with CUDA installed:

```python
import torch
import torch_conversions

print(torch_conversions.available_backends())
with torch_conversions.set_stream(device="cpu"):
    msg = torch_conversions.allocate_tensor_msg(
        (480, 640, 3), torch.uint8, device="cpu")
    output = torch_conversions.from_output_tensor_msg(msg)
    output.fill_(42)
    del output
```

```cpp
#include "torch_conversions/torch_conversions.hpp"

const auto backends = torch_conversions::available_backends();
const c10::Device device(c10::kCPU);
auto guard = torch_conversions::set_stream(device);
auto msg = torch_conversions::allocate_tensor_msg({480, 640, 3}, torch::kUInt8, device);
{
  auto output = torch_conversions::from_output_tensor_msg(*msg);
  output.fill_(42);
}
```

For CUDA, use `device="cuda:0"` in Python or `c10::Device(c10::kCUDA, 0)` in C++
with the corresponding plugin and device available. An explicit unsupported
device raises an error; it does not silently fall back to CPU. Device indices
must also be supported by the buffer allocator.

Selection depends on the operation:

| Operation | Plugin selection |
| --- | --- |
| `allocate_tensor_msg` / `set_stream` without a device | Highest priority available plugin |
| `allocate_tensor_msg` / `set_stream` with a device | Plugin registered for that Torch device type |
| `from_input_tensor_msg` / `from_output_tensor_msg` | Backend of the message storage, regardless of the default |
| `to_tensor_msg(tensor)` | Allocate using the source tensor's device, then copy |
| `to_tensor_msg(msg, tensor)` | Destination storage backend for a CPU source; source device plugin for an accelerator source |

Copies into an existing message require enough storage and a source/destination
combination supported by the selected plugin. Choosing a default accelerator does
not migrate existing messages or CPU tensors to it.

Keep message storage alive and unchanged in size while using borrowed views.
Treat input views created with `clone=False` / `clone=false` as read-only; leave
cloning enabled when you need an independent tensor. Release output views before
publication and do not modify the storage afterward. Run work on the stream used
for the view, or explicitly synchronize other streams. `set_stream` selects a
stream for the scope; it does not synchronize unrelated producer streams.

## Requirements and package structure

### Requirements

| Component | Debian packages | Source builds |
| --- | --- | --- |
| Ubuntu / ROS | Ubuntu **26.04** (Resolute), with ROS 2 Lyrical. | A ROS 2 Lyrical environment. On Ubuntu **24.04** (Noble), build ROS 2 Lyrical from source. |
| PyTorch / LibTorch | Version **2.14.0**, supplied by the ROS vendor packages. | Reuses a compatible installation **>=2.5.0**; otherwise downloads **2.14.0**. CUDA-enabled Torch downloads use `cu126` with toolkit **12.6–12.x**, or `cu130` with toolkit **13.0+**. The optional `cu132` Torch build requires toolkit **13.2+**. |
| CUDA Toolkit (example plugin) | The `cuda-toolkit` dependency, installed through APT on Resolute. | An installed toolkit compatible with the selected Torch/LibTorch build; **CUDA >=12.6** for the pinned fallback. |

For source builds with the CUDA plugin, `rosdep` installs `cuda-toolkit`
(**CUDA 13.1+**) on Resolute, but `nvidia-cuda-toolkit` (**CUDA 12.0**) on Noble,
which does not meet the pinned Torch fallback's requirement. On Noble, manually
install a compatible CUDA toolkit (**>=12.6**) and skip rosdep's CUDA installation
with `rosdep install --skip-keys cuda-toolkit`.

For Ubuntu 24.04 source builds, see the
[ROS 2 Lyrical platform requirements](https://github.com/ros2/ros2_documentation/blob/lyrical/source/Releases/lyrical/supported-platforms.rst).

### Package structure

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `torch_conversions` | `torch_conversions_py` |
| CPU plugin | `torch_conversions_cpu` | `torch_conversions_py_cpu` |
| CUDA plugin | `torch_conversions_cuda` | `torch_conversions_py_cuda` |

The core packages install example nodes from their `examples/` directories.
The CPU and CUDA plugin packages contain the backend-dependent unit and
publication tests, which use the same example nodes.

## Developing a plugin

Create separate ROS packages for the languages you support, for example
`torch_conversions_acme` and `torch_conversions_py_acme`. Use the existing
[CPU C++](torch_conversions_cpu) and [CPU Python](torch_conversions_py_cpu)
packages as minimal references, and the [CUDA C++](torch_conversions_cuda) and
[CUDA Python](torch_conversions_py_cuda) packages for device memory and streams.
The names below are placeholders for your implementation.

### Implement the storage and execution contract

Choose a unique backend identifier that matches the underlying `rosidl::Buffer`
backend type, and a device type supported by your Torch build. Report availability
only when the required runtime and hardware are usable. Priority controls default
selection (CPU uses `0`, CUDA uses `100`); it is not a mechanism for overriding
another plugin registered for the same device type. Avoid duplicate backend and
device registrations.

Your plugin must:

- Allocate the requested byte count on the requested device, including handling
  empty allocations and rejecting unsupported device indices.
- Create views with the supplied dtype, shape, element strides, and byte offset.
  Input conversion must honor the clone flag; output conversion must expose
  writable message storage.
- Retain the buffer access handle/lease for as long as a tensor view uses it.
  Acquire the appropriate read or write handle and release it through the tensor's
  ownership mechanism. Do not close a device mapping immediately after creating
  a tensor that refers to it.
- Coordinate the selected execution stream with the buffer's read/write
  synchronization. Honor an explicitly supplied stream and use the current
  device stream when none is supplied. Stream scopes must restore the previous
  stream on exit; host implementations can use a no-op scope.
- Copy non-contiguous source tensors correctly and complete `copy_to` before
  returning, as required by the public `to_tensor_msg` API. Reject unsupported
  device combinations instead of interpreting a device pointer as host memory.

The core validates message metadata before view conversion. C++ plugins receive
the message and can use `scalar_type` and `normalized_strides`; Python plugins
receive validated `TensorMetadata`. The plugin remains responsible for device
compatibility, ownership, and synchronization.

### C++: implement and export a pluginlib class

Derive a default-constructible class from
[`torch_conversions::ConversionPlugin`](torch_conversions/include/torch_conversions/conversion_plugin.hpp).
Implement all methods: `backend`, `device_type`, `available`, `priority`,
`select_stream`, `allocate`, `from_input`, `from_output`, and `copy_to`.
`select_stream` returns an optional `c10::Stream` (`std::nullopt` for a host no-op);
conversion methods receive an optional native stream through `void *`.

Export the class from its implementation file:

```cpp
#include <pluginlib/class_list_macros.hpp>
#include "torch_conversions/conversion_plugin.hpp"

// Define acme::ConversionPlugin, implementing the interface, above this macro.
PLUGINLIB_EXPORT_CLASS(acme::ConversionPlugin, torch_conversions::ConversionPlugin)
```

Add `torch_conversions_acme.xml`:

```xml
<library path="torch_conversions_acme">
  <class name="acme" type="acme::ConversionPlugin"
         base_class_type="torch_conversions::ConversionPlugin">
    <description>Torch conversions for Acme buffer storage.</description>
  </class>
</library>
```

Build a shared library and export the description using `ament_cmake`. This
CMake excerpt assumes the class is in `src/acme_conversion_plugin.cpp`; add your
Torch provider, buffer API, and device runtime dependencies and link targets:

```cmake
cmake_minimum_required(VERSION 3.20)
project(torch_conversions_acme)
set(CMAKE_CXX_STANDARD 20)
find_package(ament_cmake REQUIRED)
find_package(torch_conversions REQUIRED)
find_package(Torch REQUIRED)
find_package(pluginlib REQUIRED)

add_library(${PROJECT_NAME} SHARED src/acme_conversion_plugin.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
  torch torch_conversions::torch_conversions pluginlib::pluginlib)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
pluginlib_export_plugin_description_file(torch_conversions torch_conversions_acme.xml)
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
ament_package()
```

Declare corresponding dependencies in `package.xml`: `ament_cmake` as a build
tool, `torch_conversions`, `pluginlib`, your Torch provider and buffer/device APIs,
and any separately loaded transport backend as a runtime dependency. See the
[CUDA manifest](torch_conversions_cuda/package.xml) for an example.
Build against the same LibTorch version and compatible C++ ABI as the core;
`torch_conversions` exports an exact Torch version dependency.

### Python: implement and register the protocol

Implement the
[`ConversionPlugin` protocol](torch_conversions_py/torch_conversions/_plugin.py)
in your own module, such as `torch_conversions_acme._plugin`. No inheritance is
required. Supply `backends`, `device_types`, and `priority`, and implement
`is_available`, `matches`, `allocate`, `stream_context`, `from_input`,
`from_output`, and `copy_to` with the protocol's signatures.

For `rosidl_buffer.Buffer` storage, dispatch uses `data.backend_type`.
`matches` is used only for non-Buffer storage by plugins declaring
`is_fallback = True`, as the CPU plugin does for Python buffer-protocol objects.
A device plugin backed by `Buffer` can return `False` from `matches`.
`stream_context` must return a context manager; stream arguments are native
stream pointers represented as integers, or `None`.

Expose a registration function in that module:

```python
from torch_conversions._plugin import ConversionRegistry

# Define AcmeConversionPlugin with the protocol methods above.
def register(registry: ConversionRegistry) -> None:
    registry.register(AcmeConversionPlugin())
```

Create `resource/torch_conversions_py_acme` containing this single entry:

```text
torch_conversions_acme._plugin:register
```

Install the module and resource from the ROS package's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(torch_conversions_py_acme)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(torch_conversions_acme)
install(FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/torch_conversions__python_plugins)
ament_package()
```

Include `torch_conversions_acme/__init__.py`. In `package.xml`, declare
`ament_cmake` and `ament_cmake_python` as build tools and runtime dependencies on
`torch_conversions_py`, your Torch provider, and the buffer bindings/runtime.
See the [CUDA manifest](torch_conversions_py_cuda/package.xml). The ament marker
is required for discovery; installing an importable Python module alone is not
enough.

For device memory, the CUDA plugin demonstrates
[`make_dlpack_capsule`](torch_conversions_py/torch_conversions/_torch_bridge.pyi)
and `torch.utils.dlpack.from_dlpack`. Its owner object retains both the buffer and
access handle until Torch releases the tensor. Use your device's DLPack device
type and preserve that ownership and stream contract when adapting this pattern.

### Package and validate

Package the device runtime, Torch provider, buffer backend, and conversion plugin
as dependencies so users can install the plugin without changing application
code. The existing CPU providers are `libtorch_vendor` and
`python3_torch_vendor`; the CUDA providers are `libtorch_cuda_vendor` and
`python3_torch_cuda_vendor`. A new accelerator needs a provider with its Torch
device support enabled; a conversion plugin cannot supply that support alone.

Build your package with `colcon build --packages-up-to <plugin_package>`, source
`install/setup.bash`, and start a new process. Check `available_backends()`,
`backend_for_device(...)`, and `default_backend()` through the public API. C++
loads plugins on first registry use and skips plugins that fail to load or report
unavailability. Python loads at import time, skips modules raising `ImportError`
or `OSError`, and excludes plugins reporting unavailability; invalid resource
entries, duplicate backend names, and other registration errors can fail import.

Use the existing plugin tests as templates. Cover discovery in a fresh process,
unavailable hardware with CPU still usable, explicit device selection, allocation,
input cloning and views, output views, non-contiguous copies, metadata and device
errors, view lifetime, and stream synchronization. Run publisher/subscriber tests
across processes to verify that the buffer backend actually transports the storage
and that values remain correct after the publisher releases its local references.

## License

Apache-2.0
