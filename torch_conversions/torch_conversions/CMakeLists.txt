cmake_minimum_required(VERSION 3.20)
project(torch_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(dlpack_conversions REQUIRED)

# Header-only adapter: LibTorch is resolved in the consumer's build, so this
# package compiles without it.
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME} INTERFACE
  dlpack_conversions::dlpack_conversions
)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} NAMESPACE ${PROJECT_NAME}::)
ament_export_dependencies(dlpack_conversions)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(libtorch_vendor QUIET)
  find_package(Torch REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(rosidl_runtime_cpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(tensor_msgs REQUIRED)

  add_library(test_fixture INTERFACE)
  target_link_libraries(test_fixture INTERFACE ${PROJECT_NAME} torch)

  ament_add_gtest(test_torch_conversions test/test_torch_conversions.cpp)
  if(TARGET test_torch_conversions)
    target_link_libraries(test_torch_conversions test_fixture)
  endif()

  if(TARGET torch_cuda)
    ament_add_gtest(test_torch_conversions_cuda
      test/test_torch_conversions_cuda.cpp)
    if(TARGET test_torch_conversions_cuda)
      target_link_libraries(test_torch_conversions_cuda test_fixture)
    endif()
  endif()

  add_library(torch_tensor_publisher_component SHARED
    test/src/torch_tensor_publisher_node.cpp
  )
  set_target_properties(torch_tensor_publisher_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(torch_tensor_publisher_component
    test_fixture
    rclcpp::rclcpp
    rclcpp_components::component
    rosidl_runtime_cpp::rosidl_runtime_cpp
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
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
    test_fixture
    rclcpp::rclcpp
    rclcpp_components::component
    rosidl_runtime_cpp::rosidl_runtime_cpp
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
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

ament_package(
  CONFIG_EXTRAS_POST
    cmake/torch_conversions-extras.cmake
)
