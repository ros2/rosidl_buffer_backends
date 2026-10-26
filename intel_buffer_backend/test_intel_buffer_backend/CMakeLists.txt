cmake_minimum_required(VERSION 3.20)
project(test_intel_buffer_backend)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wformat -Wformat-security
    -fcf-protection=full
    -D_FORTIFY_SOURCE=3
    -D_GLIBCXX_ASSERTIONS
  )
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fstack-protector)
  else()
    add_compile_options(
      -Werror -Wconversion -Wimplicit-fallthrough
      -Werror=format-security
      -fstack-protector-strong -fstack-clash-protection
    )
    add_link_options(
      -Wl,-z,noexecstack
      -Wl,-z,relro,-z,now
    )
  endif()
endif()

find_package(ament_cmake REQUIRED)
find_package(intel_buffer REQUIRED)
find_package(intel_memory_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# --- Camera publisher (camera/video → sensor_msgs/Image backed by IntelBuffer) ---
add_library(video_publisher_component SHARED
  src/video_publisher_node.cpp
)
target_include_directories(video_publisher_component PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(video_publisher_component
  intel_buffer::intel_buffer
  intel_memory_core::intel_memory_core
  rclcpp::rclcpp
  rclcpp_components::component
  ${sensor_msgs_TARGETS}
  ${OpenCV_LIBRARIES}
)
rclcpp_components_register_node(video_publisher_component
  PLUGIN "UsmVideoPublisher"
  EXECUTABLE video_publisher_node
)

# --- OpenVINO inference (sensor_msgs/Image → IntelBuffer → GPU/NPU) ---
add_library(openvino_pipeline_component SHARED
  src/openvino_pipeline_node.cpp
)
target_include_directories(openvino_pipeline_component PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(openvino_pipeline_component
  intel_buffer::intel_buffer
  intel_memory_core::intel_memory_core
  rclcpp::rclcpp
  rclcpp_components::component
  ${sensor_msgs_TARGETS}
  ${vision_msgs_TARGETS}
  ${OpenCV_LIBRARIES}
  openvino::runtime
)
rclcpp_components_register_node(openvino_pipeline_component
  PLUGIN "UsmOpenvinoPipeline"
  EXECUTABLE openvino_pipeline_node
)

# --- Detection visualizer (draws boxes, publishes Image for RViz2) ---
add_library(detection_visualizer_component SHARED
  src/detection_visualizer_node.cpp
)
target_include_directories(detection_visualizer_component PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(detection_visualizer_component
  intel_buffer::intel_buffer
  intel_memory_core::intel_memory_core
  rclcpp::rclcpp
  rclcpp_components::component
  ${sensor_msgs_TARGETS}
  ${vision_msgs_TARGETS}
  ${OpenCV_LIBRARIES}
)
rclcpp_components_register_node(detection_visualizer_component
  PLUGIN "UsmDetectionVisualizer"
  EXECUTABLE detection_visualizer_node
)

install(TARGETS
  video_publisher_component
  openvino_pipeline_component
  detection_visualizer_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  video_publisher_node
  openvino_pipeline_node
  detection_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_ros REQUIRED)

  # --- End-to-end rosidl::Buffer test: sensor_msgs/Image through RMW/backend ---
  # Publisher runs as its own process so imports are cross-process.
  add_executable(intel_buffer_e2e_publisher test/intel_buffer_e2e_publisher.cpp)
  target_link_libraries(intel_buffer_e2e_publisher
    intel_buffer::intel_buffer
    intel_memory_core::intel_memory_core
    rclcpp::rclcpp
    ${sensor_msgs_TARGETS}
  )

  ament_add_ros_isolated_gtest(test_intel_buffer_e2e test/test_intel_buffer_e2e.cpp
    TIMEOUT 300)
  target_compile_definitions(test_intel_buffer_e2e PRIVATE
    INTEL_BUFFER_E2E_PUBLISHER="$<TARGET_FILE:intel_buffer_e2e_publisher>")
  target_link_libraries(test_intel_buffer_e2e
    intel_buffer::intel_buffer
    intel_memory_core::intel_memory_core
    rclcpp::rclcpp
    ${sensor_msgs_TARGETS}
  )
  add_dependencies(test_intel_buffer_e2e intel_buffer_e2e_publisher)
endif()

ament_package()
