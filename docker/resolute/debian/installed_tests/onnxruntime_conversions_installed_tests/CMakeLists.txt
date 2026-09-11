# Copyright 2026 Open Source Robotics Foundation, Inc.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions_installed_tests)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(launch_testing_ament_cmake REQUIRED)
find_package(onnxruntime_conversions REQUIRED)
find_package(onnxruntime_cuda_vendor REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tensor_msgs REQUIRED)

set(BACKENDS_SOURCE_ROOT "" CACHE PATH "Path to the source checkout")
set(TEST_ROOT
  "${BACKENDS_SOURCE_ROOT}/onnxruntime_conversions/onnxruntime_conversions/test")

add_library(test_fixture INTERFACE)
target_link_libraries(test_fixture INTERFACE
  onnxruntime_conversions::onnxruntime_conversions
  onnxruntime::onnxruntime)

ament_add_gtest(test_onnxruntime_conversions
  "${TEST_ROOT}/test_onnxruntime_conversions.cpp")
target_link_libraries(test_onnxruntime_conversions test_fixture)

ament_add_gtest(test_onnxruntime_conversions_cuda
  "${TEST_ROOT}/test_onnxruntime_conversions_cuda.cpp")
target_link_libraries(test_onnxruntime_conversions_cuda
  test_fixture CUDA::cudart)

add_executable(runtime_probe src/runtime_probe.cpp)
target_link_libraries(runtime_probe test_fixture)

add_library(ort_tensor_publisher_component SHARED
  "${TEST_ROOT}/src/ort_tensor_publisher_node.cpp")
target_link_libraries(ort_tensor_publisher_component
  test_fixture CUDA::cudart rclcpp::rclcpp rclcpp_components::component
  rosidl_runtime_cpp::rosidl_runtime_cpp ${std_msgs_TARGETS} ${tensor_msgs_TARGETS})
rclcpp_components_register_node(ort_tensor_publisher_component
  PLUGIN "OrtTensorPublisher" EXECUTABLE ort_tensor_publisher_node)

add_library(ort_tensor_subscriber_component SHARED
  "${TEST_ROOT}/src/ort_tensor_subscriber_node.cpp")
target_link_libraries(ort_tensor_subscriber_component
  test_fixture CUDA::cudart rclcpp::rclcpp rclcpp_components::component
  rosidl_runtime_cpp::rosidl_runtime_cpp ${std_msgs_TARGETS} ${tensor_msgs_TARGETS})
rclcpp_components_register_node(ort_tensor_subscriber_component
  PLUGIN "OrtTensorSubscriber" EXECUTABLE ort_tensor_subscriber_node)

set(launch_file test_ort_tensor_inter_pubsub_fastrtps_launch.py)
file(READ "${TEST_ROOT}/${launch_file}" content)
string(REPLACE "package='onnxruntime_conversions'"
  "package='onnxruntime_conversions_installed_tests'" content "${content}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${launch_file}" "${content}")

install(TARGETS
    test_onnxruntime_conversions
    test_onnxruntime_conversions_cuda
    runtime_probe
    ort_tensor_publisher_component
    ort_tensor_subscriber_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${launch_file}"
  DESTINATION share/${PROJECT_NAME}/test)

ament_package()
