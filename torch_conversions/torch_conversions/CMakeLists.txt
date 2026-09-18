cmake_minimum_required(VERSION 3.20)
project(torch_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tensor_msgs REQUIRED)
find_package(rmw REQUIRED)
find_package(rcutils REQUIRED)

find_package(cuda_buffer QUIET)
find_package(libtorch_vendor QUIET)

find_package(Torch REQUIRED)

set(_torch_has_cuda FALSE)
if(TARGET torch_cuda)
  set(_torch_has_cuda TRUE)
  find_package(CUDAToolkit QUIET)
endif()

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  INTERFACE
    ${TORCH_LIBRARIES}
    rosidl_buffer::rosidl_buffer
    rmw::rmw
    rcutils::rcutils
    ${sensor_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
)

if(_torch_has_cuda AND CUDAToolkit_FOUND)
  target_link_libraries(${PROJECT_NAME} INTERFACE CUDA::cudart)
  target_compile_definitions(${PROJECT_NAME} INTERFACE
    __NV_NO_HOST_COMPILER_CHECK=1)
endif()

if(cuda_buffer_FOUND)
  target_link_libraries(${PROJECT_NAME} INTERFACE cuda_buffer::cuda_buffer)
endif()

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(${PROJECT_NAME})
ament_export_dependencies(
  libtorch_vendor rosidl_buffer sensor_msgs tensor_msgs rmw rcutils)
if(_torch_has_cuda AND CUDAToolkit_FOUND)
  ament_export_dependencies(CUDAToolkit)
endif()
if(cuda_buffer_FOUND)
  ament_export_dependencies(cuda_buffer)
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_add_gtest(test_torch_conversions
    test/test_torch_conversions.cpp
  )
  if(TARGET test_torch_conversions)
    target_link_libraries(test_torch_conversions
      ${PROJECT_NAME}
      ${TORCH_LIBRARIES}
      rosidl_buffer::rosidl_buffer
      ${sensor_msgs_TARGETS}
      ${tensor_msgs_TARGETS}
    )
    if(_torch_has_cuda AND CUDAToolkit_FOUND)
      # CudaError uses cuGetErrorName/cuGetErrorString from the CUDA driver lib
      target_link_libraries(test_torch_conversions CUDA::cuda_driver)
    endif()
  endif()

  add_library(torch_tensor_publisher_component SHARED
    test/src/torch_tensor_publisher_node.cpp
  )
  set_target_properties(torch_tensor_publisher_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(torch_tensor_publisher_component
    ${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    rclcpp::rclcpp
    rclcpp_components::component
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
    rosidl_runtime_cpp::rosidl_runtime_cpp
  )
  if(_torch_has_cuda AND CUDAToolkit_FOUND)
    target_link_libraries(torch_tensor_publisher_component CUDA::cuda_driver)
  endif()
  rclcpp_components_register_node(torch_tensor_publisher_component
    PLUGIN "TorchTensorPublisher"
    EXECUTABLE torch_tensor_publisher_node
  )
  install(TARGETS torch_tensor_publisher_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  add_library(torch_tensor_subscriber_component SHARED
    test/src/torch_tensor_subscriber_node.cpp
  )
  set_target_properties(torch_tensor_subscriber_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(torch_tensor_subscriber_component
    ${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    rclcpp::rclcpp
    rclcpp_components::component
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
    rosidl_runtime_cpp::rosidl_runtime_cpp
  )
  if(_torch_has_cuda AND CUDAToolkit_FOUND)
    target_link_libraries(torch_tensor_subscriber_component CUDA::cuda_driver)
  endif()
  rclcpp_components_register_node(torch_tensor_subscriber_component
    PLUGIN "TorchTensorSubscriber"
    EXECUTABLE torch_tensor_subscriber_node
  )
  install(TARGETS torch_tensor_subscriber_component
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
