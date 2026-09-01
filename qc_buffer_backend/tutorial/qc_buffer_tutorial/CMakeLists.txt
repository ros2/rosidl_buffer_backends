cmake_minimum_required(VERSION 3.20)
project(qc_buffer_tutorial)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(qc_buffer REQUIRED)

add_library(qc_image_publisher_component SHARED src/qc_image_publisher.cpp)
target_link_libraries(qc_image_publisher_component
  rclcpp::rclcpp
  rclcpp_components::component
  qc_buffer::qc_buffer
  rosidl_runtime_cpp::rosidl_runtime_cpp
  ${sensor_msgs_TARGETS}
)
rclcpp_components_register_nodes(qc_image_publisher_component "QcImagePublisher")

add_library(qc_image_subscriber_component SHARED src/qc_image_subscriber.cpp)
target_link_libraries(qc_image_subscriber_component
  rclcpp::rclcpp
  rclcpp_components::component
  qc_buffer::qc_buffer
  rosidl_runtime_cpp::rosidl_runtime_cpp
  ${sensor_msgs_TARGETS}
)
rclcpp_components_register_nodes(qc_image_subscriber_component "QcImageSubscriber")

install(TARGETS
  qc_image_publisher_component
  qc_image_subscriber_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
