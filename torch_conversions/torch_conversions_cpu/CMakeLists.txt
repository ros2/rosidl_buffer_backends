cmake_minimum_required(VERSION 3.20)
project(torch_conversions_cpu)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(torch_conversions REQUIRED COMPONENTS Interface)
find_package(Torch 2.9.1 EXACT REQUIRED)

if(TARGET torch_cuda)
  message(FATAL_ERROR
    "${PROJECT_NAME} requires Ubuntu's CPU-only libtorch-dev 2.9.1")
endif()

add_library(torch_conversions_cpu_impl SHARED
  src/torch_conversions_cpu_impl.cpp
)
target_compile_definitions(torch_conversions_cpu_impl
  PRIVATE TORCH_CONVERSIONS_BUILDING_DLL)
target_link_libraries(torch_conversions_cpu_impl
  PUBLIC torch_conversions::interface torch
  PRIVATE pluginlib::pluginlib
)

add_library(torch_conversions_cpu_plugin SHARED
  src/cpu_conversion_plugin.cpp
)
target_link_libraries(torch_conversions_cpu_plugin
  PRIVATE torch_conversions::interface pluginlib::pluginlib
)

pluginlib_export_plugin_description_file(
  torch_conversions
  torch_conversions_cpu.xml
)

install(
  TARGETS torch_conversions_cpu_impl torch_conversions_cpu_plugin
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/torch_conversions__runtime
)

ament_export_targets(export_${PROJECT_NAME} NAMESPACE ${PROJECT_NAME}::)
ament_export_dependencies(pluginlib Torch)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(rosidl_runtime_cpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(tensor_msgs REQUIRED)

  ament_add_gtest(test_torch_conversions_cpu
    test/test_torch_conversions_cpu.cpp
  )
  if(TARGET test_torch_conversions_cpu)
    target_link_libraries(test_torch_conversions_cpu
      torch_conversions_cpu_impl
      torch
    )
    add_dependencies(test_torch_conversions_cpu torch_conversions_cpu_plugin)
  endif()

  add_library(torch_tensor_publisher_component SHARED
    test/src/torch_tensor_publisher_node.cpp
  )
  set_target_properties(torch_tensor_publisher_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(torch_tensor_publisher_component
    torch_conversions_cpu_impl
    torch
    rclcpp::rclcpp
    rclcpp_components::component
    rosidl_runtime_cpp::rosidl_runtime_cpp
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
  )
  add_dependencies(
    torch_tensor_publisher_component
    torch_conversions_cpu_plugin
  )
  rclcpp_components_register_node(torch_tensor_publisher_component
    PLUGIN "TorchTensorPublisher"
    EXECUTABLE torch_tensor_publisher_node
  )

  add_library(torch_tensor_subscriber_component SHARED
    test/src/torch_tensor_subscriber_node.cpp
  )
  set_target_properties(torch_tensor_subscriber_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(torch_tensor_subscriber_component
    torch_conversions_cpu_impl
    torch
    rclcpp::rclcpp
    rclcpp_components::component
    rosidl_runtime_cpp::rosidl_runtime_cpp
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
  )
  add_dependencies(
    torch_tensor_subscriber_component
    torch_conversions_cpu_plugin
  )
  rclcpp_components_register_node(torch_tensor_subscriber_component
    PLUGIN "TorchTensorSubscriber"
    EXECUTABLE torch_tensor_subscriber_node
  )

  install(
    TARGETS
      torch_tensor_publisher_component
      torch_tensor_subscriber_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  add_launch_test(test/test_torch_tensor_intra_pubsub_fastrtps_launch.py
    TARGET test_torch_tensor_intra_pubsub_fastrtps_launch
    TIMEOUT 30
  )
  add_launch_test(test/test_torch_tensor_inter_pubsub_fastrtps_launch.py
    TARGET test_torch_tensor_inter_pubsub_fastrtps_launch
    TIMEOUT 30
  )
  add_launch_test(test/test_torch_tensor_multi_pub_fastrtps_launch.py
    TARGET test_torch_tensor_multi_pub_fastrtps_launch
    TIMEOUT 45
  )
  add_launch_test(test/test_torch_tensor_multi_sub_fastrtps_launch.py
    TARGET test_torch_tensor_multi_sub_fastrtps_launch
    TIMEOUT 45
  )
endif()

ament_package()
