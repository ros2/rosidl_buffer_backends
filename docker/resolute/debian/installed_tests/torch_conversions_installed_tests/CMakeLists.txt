# Copyright 2026 Open Source Robotics Foundation, Inc.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(torch_conversions_installed_tests)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(launch_testing_ament_cmake REQUIRED)
option(BUILD_CUDA_TESTS
  "Compile tests that require the optional CUDA LibTorch provider" OFF)
if(BUILD_CUDA_TESTS)
  find_package(libtorch_cuda_vendor REQUIRED)
else()
  find_package(libtorch_vendor REQUIRED)
endif()
find_package(Torch REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tensor_msgs REQUIRED)
find_package(torch_conversions REQUIRED)

set(BACKENDS_SOURCE_ROOT "" CACHE PATH
  "Path to the rosidl_buffer_backends source checkout")
if(BACKENDS_SOURCE_ROOT STREQUAL "")
  get_filename_component(BACKENDS_SOURCE_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
endif()
set(CPP_TEST_ROOT
  "${BACKENDS_SOURCE_ROOT}/torch_conversions/torch_conversions/test")
set(PY_TEST_ROOT
  "${BACKENDS_SOURCE_ROOT}/torch_conversions/torch_conversions_py/test")

ament_add_gtest(test_torch_conversions
  "${CPP_TEST_ROOT}/test_torch_conversions.cpp")
target_link_libraries(test_torch_conversions torch_conversions::torch_conversions)

if(BUILD_CUDA_TESTS)
  ament_add_gtest(test_torch_conversions_cuda
    "${CPP_TEST_ROOT}/test_torch_conversions_cuda.cpp")
  target_link_libraries(test_torch_conversions_cuda
    torch_conversions::torch_conversions)
endif()

add_executable(runtime_probe src/runtime_probe.cpp)
target_link_libraries(runtime_probe torch_conversions::torch_conversions)

add_library(torch_tensor_publisher_component SHARED
  "${CPP_TEST_ROOT}/src/torch_tensor_publisher_node.cpp")
target_link_libraries(torch_tensor_publisher_component
  torch_conversions::torch_conversions
  rclcpp::rclcpp
  rclcpp_components::component
  rosidl_runtime_cpp::rosidl_runtime_cpp
  ${std_msgs_TARGETS}
  ${tensor_msgs_TARGETS})
rclcpp_components_register_node(torch_tensor_publisher_component
  PLUGIN "TorchTensorPublisher"
  EXECUTABLE torch_tensor_publisher_node)

add_library(torch_tensor_subscriber_component SHARED
  "${CPP_TEST_ROOT}/src/torch_tensor_subscriber_node.cpp")
target_link_libraries(torch_tensor_subscriber_component
  torch_conversions::torch_conversions
  rclcpp::rclcpp
  rclcpp_components::component
  rosidl_runtime_cpp::rosidl_runtime_cpp
  ${std_msgs_TARGETS}
  ${tensor_msgs_TARGETS})
rclcpp_components_register_node(torch_tensor_subscriber_component
  PLUGIN "TorchTensorSubscriber"
  EXECUTABLE torch_tensor_subscriber_node)

foreach(name IN ITEMS
    test_torch_tensor_intra_pubsub_fastrtps_launch.py
    test_torch_tensor_inter_pubsub_fastrtps_launch.py
    test_torch_tensor_multi_pub_fastrtps_launch.py
    test_torch_tensor_multi_sub_fastrtps_launch.py)
  file(READ "${CPP_TEST_ROOT}/${name}" content)
  string(REPLACE "package='torch_conversions'"
    "package='torch_conversions_installed_tests'" content "${content}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${name}" "${content}")
endforeach()

set(python_launch test_torch_tensor_inter_pubsub_fastrtps.py)
file(READ "${PY_TEST_ROOT}/${python_launch}" content)
string(REPLACE "package='torch_conversions_py'"
  "package='torch_conversions_installed_tests'" content "${content}")
string(REPLACE "executable='torch_tensor_publisher_node'"
  "executable='torch_tensor_publisher_node_py'" content "${content}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${python_launch}" "${content}")

set(_installed_test_targets
    test_torch_conversions
    runtime_probe
    torch_tensor_publisher_component
    torch_tensor_subscriber_component)
if(BUILD_CUDA_TESTS)
  list(APPEND _installed_test_targets test_torch_conversions_cuda)
endif()
install(TARGETS ${_installed_test_targets}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS "${PY_TEST_ROOT}/torch_tensor_publisher_node.py"
  DESTINATION lib/${PROJECT_NAME}
  RENAME torch_tensor_publisher_node_py)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/test_torch_tensor_intra_pubsub_fastrtps_launch.py"
    "${CMAKE_CURRENT_BINARY_DIR}/test_torch_tensor_inter_pubsub_fastrtps_launch.py"
    "${CMAKE_CURRENT_BINARY_DIR}/test_torch_tensor_multi_pub_fastrtps_launch.py"
    "${CMAKE_CURRENT_BINARY_DIR}/test_torch_tensor_multi_sub_fastrtps_launch.py"
    "${CMAKE_CURRENT_BINARY_DIR}/${python_launch}"
  DESTINATION share/${PROJECT_NAME}/test)

ament_package()
