cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions_cuda)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(cuda_buffer REQUIRED)
find_package(onnxruntime_conversions REQUIRED)
find_package(onnxruntime_cuda_vendor REQUIRED)
find_package(pluginlib REQUIRED)

set(_cuda_provider_library "${onnxruntime_cuda_vendor_LIBRARY}")
if(NOT EXISTS "${_cuda_provider_library}")
  get_filename_component(
    _cuda_provider_prefix
    "${onnxruntime_cuda_vendor_DIR}/../../.."
    ABSOLUTE
  )
  set(_cuda_provider_library
    "${_cuda_provider_prefix}/opt/onnxruntime_core_vendor/lib/libonnxruntime_providers_cuda.so"
  )
endif()
if(NOT EXISTS "${_cuda_provider_library}")
  message(FATAL_ERROR
    "${PROJECT_NAME}: CUDA execution-provider library was not installed by "
    "onnxruntime_cuda_vendor"
  )
endif()

add_library(${PROJECT_NAME} SHARED
  src/cuda_conversion_backend.cpp
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    onnxruntime_conversions::onnxruntime_conversions
    onnxruntime::onnxruntime
    cuda_buffer::cuda_buffer
    CUDA::cudart
    CUDA::cuda_driver
    pluginlib::pluginlib
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

pluginlib_export_plugin_description_file(
  onnxruntime_conversions
  onnxruntime_conversions_cuda.xml
)

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_onnxruntime_conversions_cuda
    test/test_cuda_conversion_backend.cpp
  )
  if(TARGET test_onnxruntime_conversions_cuda)
    target_link_libraries(test_onnxruntime_conversions_cuda
      onnxruntime_conversions::onnxruntime_conversions
      onnxruntime::onnxruntime
      cuda_buffer::cuda_buffer
      CUDA::cudart
      CUDA::cuda_driver
      pluginlib::pluginlib
    )
    add_dependencies(test_onnxruntime_conversions_cuda ${PROJECT_NAME})
    set_target_properties(test_onnxruntime_conversions_cuda PROPERTIES
      BUILD_RPATH
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>;$<TARGET_FILE_DIR:onnxruntime::onnxruntime>"
    )
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(tensor_msgs REQUIRED)

  add_library(onnxruntime_cuda_tensor_publisher_component SHARED
    test/src/cuda_tensor_publisher_node.cpp
  )
  target_link_libraries(onnxruntime_cuda_tensor_publisher_component
    onnxruntime_conversions::onnxruntime_conversions
    onnxruntime::onnxruntime
    rclcpp::rclcpp
    rclcpp_components::component
    CUDA::cudart
    CUDA::cuda_driver
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
  )
  add_dependencies(onnxruntime_cuda_tensor_publisher_component ${PROJECT_NAME})
  rclcpp_components_register_node(onnxruntime_cuda_tensor_publisher_component
    PLUGIN "CudaTensorPublisher"
    EXECUTABLE onnxruntime_cuda_tensor_publisher_node
  )

  add_library(onnxruntime_cuda_tensor_subscriber_component SHARED
    test/src/cuda_tensor_subscriber_node.cpp
  )
  target_link_libraries(onnxruntime_cuda_tensor_subscriber_component
    onnxruntime_conversions::onnxruntime_conversions
    onnxruntime::onnxruntime
    rclcpp::rclcpp
    rclcpp_components::component
    CUDA::cudart
    CUDA::cuda_driver
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
  )
  add_dependencies(onnxruntime_cuda_tensor_subscriber_component ${PROJECT_NAME})
  rclcpp_components_register_node(onnxruntime_cuda_tensor_subscriber_component
    PLUGIN "CudaTensorSubscriber"
    EXECUTABLE onnxruntime_cuda_tensor_subscriber_node
  )

  install(
    TARGETS
      onnxruntime_cuda_tensor_publisher_component
      onnxruntime_cuda_tensor_subscriber_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  add_launch_test(
    test/test_cuda_tensor_inter_pubsub_fastrtps_launch.py
    TARGET test_cuda_tensor_inter_pubsub_fastrtps
    TIMEOUT 40
  )
endif()

ament_package()
