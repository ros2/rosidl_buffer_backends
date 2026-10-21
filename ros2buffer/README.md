# ros2buffer

`ros2buffer` provides a separately maintained ROS 2 CLI command for
inspecting ROSIDL buffer backend support:

```bash
ros2 buffer info --no-daemon /topic_name
```

The command resolves the topic through the ROS graph, lists its publishers and
subscribers separately, and identifies every endpoint by node name, GID, and
topic type.

## Fast DDS backend metadata

When `rmw_fastrtps_cpp` provides the implementation-specific
`rmw_fastrtps_cpp_get_endpoint_buffer_backend_metadata` function, the command
dynamically loads it and reports the backend-to-metadata map advertised by each
publisher or subscriber. `--no-daemon` is recommended so graph discovery and
the implementation-specific metadata registry belong to the same CLI process.

The symbol is resolved at runtime; `ros2buffer` therefore remains loadable with
an older `rmw_fastrtps_cpp`. Instead of crashing, each affected endpoint is
reported with a message similar to:

```text
Buffer backends: unavailable (incompatible rmw_fastrtps_cpp: the installed version does not expose rmw_fastrtps_cpp_get_endpoint_buffer_backend_metadata)
```

Other RMW implementations are also handled without failure and reported as
unsupported by this experimental version. A stable public graph API will still
be needed before this mechanism can support Fast DDS and Zenoh uniformly.

## Live demo

Build the command, launch extension, and demo application from the aggregate
workspace:

```bash
pixi run build "test_rosidl_buffer_nv ros2launch ros2buffer"
```

Then enter `pixi shell`, source `install/setup.sh`, and run the demo in one
terminal:

```bash
RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ros2 launch test_rosidl_buffer_nv demo_image_60fps.launch.py
```

In a second configured terminal, inspect its topic directly:

```bash
RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ros2 buffer info --no-daemon /demo_image_60fps
```
