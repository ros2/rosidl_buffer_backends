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
find_package(onnxruntime_conversions REQUIRED COMPONENTS Interface)
find_package(onnxruntime_cuda_vendor REQUIRED)
find_package(pluginlib REQUIRED)

if(NOT CUDAToolkit_VERSION_MAJOR EQUAL 12)
  message(FATAL_ERROR
    "${PROJECT_NAME}: ONNX Runtime 1.23.2 requires CUDA 12, "
    "but found CUDA ${CUDAToolkit_VERSION}")
endif()

set(_cuda_provider_library "${onnxruntime_cuda_vendor_LIBRARY}")
if(NOT EXISTS "${_cuda_provider_library}")
  message(FATAL_ERROR
    "${PROJECT_NAME}: CUDA execution-provider library was not installed by "
    "onnxruntime_cuda_vendor"
  )
endif()

add_library(${PROJECT_NAME}_impl SHARED
  src/onnxruntime_conversions_cuda_impl.cpp)
target_compile_definitions(${PROJECT_NAME}_impl
  PRIVATE ONNXRUNTIME_CONVERSIONS_BUILDING_DLL)
target_compile_features(${PROJECT_NAME}_impl PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_impl
  PUBLIC
    onnxruntime_conversions::interface
    onnxruntime::onnxruntime
  PRIVATE
    ${CMAKE_DL_LIBS}
)

add_library(${PROJECT_NAME}_plugin SHARED
  src/cuda_conversion_plugin.cpp)
target_compile_features(${PROJECT_NAME}_plugin PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_plugin
  PRIVATE
    ${PROJECT_NAME}_impl
    onnxruntime::onnxruntime
    cuda_buffer::cuda_buffer
    CUDA::cudart
    pluginlib::pluginlib
)
pluginlib_export_plugin_description_file(
  onnxruntime_conversions
  onnxruntime_conversions_cuda.xml
)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/onnxruntime_conversions__runtime
)
install(
  TARGETS ${PROJECT_NAME}_impl
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  TARGETS ${PROJECT_NAME}_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  HAS_LIBRARY_TARGET
)
ament_export_dependencies(
  onnxruntime_cuda_vendor
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_onnxruntime_conversions_cuda
    test/test_cuda_conversion_plugin.cpp
  )
  if(TARGET test_onnxruntime_conversions_cuda)
    target_link_libraries(test_onnxruntime_conversions_cuda
      ${PROJECT_NAME}_impl
      onnxruntime::onnxruntime
      cuda_buffer::cuda_buffer
      CUDA::cudart
      pluginlib::pluginlib
    )
    add_dependencies(
      test_onnxruntime_conversions_cuda
      ${PROJECT_NAME}_plugin
    )
    set_target_properties(test_onnxruntime_conversions_cuda PROPERTIES
      BUILD_RPATH
        "$<TARGET_FILE_DIR:${PROJECT_NAME}_plugin>;$<TARGET_FILE_DIR:onnxruntime::onnxruntime>"
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
    ${PROJECT_NAME}_impl
    onnxruntime::onnxruntime
    rclcpp::rclcpp
    rclcpp_components::component
    CUDA::cudart
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
  )
  add_dependencies(
    onnxruntime_cuda_tensor_publisher_component
    ${PROJECT_NAME}_plugin
  )
  rclcpp_components_register_node(onnxruntime_cuda_tensor_publisher_component
    PLUGIN "CudaTensorPublisher"
    EXECUTABLE onnxruntime_cuda_tensor_publisher_node
  )

  add_library(onnxruntime_cuda_tensor_subscriber_component SHARED
    test/src/cuda_tensor_subscriber_node.cpp
  )
  target_link_libraries(onnxruntime_cuda_tensor_subscriber_component
    ${PROJECT_NAME}_impl
    onnxruntime::onnxruntime
    rclcpp::rclcpp
    rclcpp_components::component
    CUDA::cudart
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
  )
  add_dependencies(
    onnxruntime_cuda_tensor_subscriber_component
    ${PROJECT_NAME}_plugin
  )
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
