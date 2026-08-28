cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(ONNXRUNTIME REQUIRED MODULE)
find_package(rosidl_buffer REQUIRED)
find_package(tensor_msgs REQUIRED)
find_package(cuda_buffer QUIET)

get_target_property(_onnxruntime_library
  onnxruntime::onnxruntime IMPORTED_LOCATION)
get_filename_component(_onnxruntime_library_dir
  "${_onnxruntime_library}" DIRECTORY)
find_library(_onnxruntime_cuda_provider
  NAMES onnxruntime_providers_cuda
  HINTS "${_onnxruntime_library_dir}"
  NO_DEFAULT_PATH
)

add_library(${PROJECT_NAME} SHARED
  src/onnxruntime_conversions.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    onnxruntime::onnxruntime
    rosidl_buffer::rosidl_buffer
    ${tensor_msgs_TARGETS}
)

if(cuda_buffer_FOUND)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(${PROJECT_NAME}
    PRIVATE ONNXRUNTIME_CONVERSIONS_HAS_CUDA)
  target_link_libraries(${PROJECT_NAME}
    PRIVATE cuda_buffer::cuda_buffer CUDA::cudart)
endif()

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)
install(
  DIRECTORY cmake/
  DESTINATION share/${PROJECT_NAME}/cmake/modules
)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rosidl_buffer tensor_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_onnxruntime_conversions
    test/test_onnxruntime_conversions.cpp
  )
  if(TARGET test_onnxruntime_conversions)
    target_link_libraries(test_onnxruntime_conversions
      ${PROJECT_NAME}
      onnxruntime::onnxruntime
      rosidl_buffer::rosidl_buffer
      ${tensor_msgs_TARGETS}
    )
    if(cuda_buffer_FOUND)
      target_compile_definitions(test_onnxruntime_conversions
        PRIVATE ONNXRUNTIME_CONVERSIONS_TEST_HAS_CUDA)
      target_link_libraries(test_onnxruntime_conversions
        cuda_buffer::cuda_buffer CUDA::cudart)
      if(_onnxruntime_cuda_provider)
        target_compile_definitions(test_onnxruntime_conversions
          PRIVATE ONNXRUNTIME_CONVERSIONS_TEST_HAS_CUDA_EP)
      endif()
    endif()
  endif()

  if(cuda_buffer_FOUND AND _onnxruntime_cuda_provider)
    find_package(launch_testing_ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(rclcpp_components REQUIRED)
    find_package(std_msgs REQUIRED)

    add_library(onnxruntime_tensor_publisher_component SHARED
      test/src/onnxruntime_tensor_publisher_node.cpp
    )
    set_target_properties(onnxruntime_tensor_publisher_component PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE)
    target_link_libraries(onnxruntime_tensor_publisher_component
      ${PROJECT_NAME}
      onnxruntime::onnxruntime
      rclcpp::rclcpp
      rclcpp_components::component
      CUDA::cudart
      ${std_msgs_TARGETS}
      ${tensor_msgs_TARGETS}
    )
    rclcpp_components_register_node(onnxruntime_tensor_publisher_component
      PLUGIN "OnnxRuntimeTensorPublisher"
      EXECUTABLE onnxruntime_tensor_publisher_node
    )

    add_library(onnxruntime_tensor_subscriber_component SHARED
      test/src/onnxruntime_tensor_subscriber_node.cpp
    )
    set_target_properties(onnxruntime_tensor_subscriber_component PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE)
    target_link_libraries(onnxruntime_tensor_subscriber_component
      ${PROJECT_NAME}
      onnxruntime::onnxruntime
      rclcpp::rclcpp
      rclcpp_components::component
      CUDA::cudart
      ${std_msgs_TARGETS}
      ${tensor_msgs_TARGETS}
    )
    rclcpp_components_register_node(onnxruntime_tensor_subscriber_component
      PLUGIN "OnnxRuntimeTensorSubscriber"
      EXECUTABLE onnxruntime_tensor_subscriber_node
    )

    install(
      TARGETS
        onnxruntime_tensor_publisher_component
        onnxruntime_tensor_subscriber_component
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin
    )

    add_launch_test(
      test/test_onnxruntime_tensor_inter_pubsub_fastrtps_launch.py
      TARGET test_onnxruntime_tensor_inter_pubsub_fastrtps
      TIMEOUT 40
    )
  endif()
endif()

ament_package(CONFIG_EXTRAS "onnxruntime_conversions-extras.cmake")
