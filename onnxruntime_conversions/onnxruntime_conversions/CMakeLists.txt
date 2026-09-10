cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(dlpack_conversions REQUIRED)

# Header-only adapter: ONNX Runtime is resolved in the consumer's build, so
# this package compiles without it.
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
  # The CUDA tests and component nodes drive the runtime API directly, so they
  # need the toolkit even though the adapter itself does not.
  find_package(CUDAToolkit QUIET)
  # Either vendor defines onnxruntime::onnxruntime; the CUDA one is preferred
  # because it is the superset. A plain find_package(onnxruntime) covers an
  # externally provided SDK.
  find_package(onnxruntime_cuda_vendor QUIET)
  if(NOT TARGET onnxruntime::onnxruntime)
    find_package(onnxruntime_core_vendor QUIET)
  endif()
  if(NOT TARGET onnxruntime::onnxruntime)
    find_package(onnxruntime REQUIRED)
  endif()
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(rosidl_runtime_cpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(tensor_msgs REQUIRED)

  add_library(test_fixture INTERFACE)
  target_link_libraries(test_fixture INTERFACE
    ${PROJECT_NAME}
    onnxruntime::onnxruntime
  )

  ament_add_gtest(test_onnxruntime_conversions
    test/test_onnxruntime_conversions.cpp)
  if(TARGET test_onnxruntime_conversions)
    target_link_libraries(test_onnxruntime_conversions test_fixture)
  endif()

  # The pub/sub nodes exchange device storage and ask ONNX Runtime for the CUDA
  # execution provider, so they and the CUDA test need the toolkit and an ONNX
  # Runtime that actually carries that provider. A CPU-only ONNX Runtime on a
  # CUDA machine satisfies the toolkit check alone and then fails at runtime.
  if(CUDAToolkit_FOUND AND onnxruntime_cuda_vendor_FOUND)
    ament_add_gtest(test_onnxruntime_conversions_cuda
      test/test_onnxruntime_conversions_cuda.cpp)
    if(TARGET test_onnxruntime_conversions_cuda)
      target_link_libraries(test_onnxruntime_conversions_cuda
        test_fixture CUDA::cudart)
    endif()

    add_library(ort_tensor_publisher_component SHARED
      test/src/ort_tensor_publisher_node.cpp
    )
    set_target_properties(ort_tensor_publisher_component PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE)
    target_link_libraries(ort_tensor_publisher_component
      test_fixture
      CUDA::cudart
      rclcpp::rclcpp
      rclcpp_components::component
      rosidl_runtime_cpp::rosidl_runtime_cpp
      ${std_msgs_TARGETS}
      ${tensor_msgs_TARGETS}
    )
    rclcpp_components_register_node(ort_tensor_publisher_component
      PLUGIN "OrtTensorPublisher"
      EXECUTABLE ort_tensor_publisher_node
    )

    add_library(ort_tensor_subscriber_component SHARED
      test/src/ort_tensor_subscriber_node.cpp
    )
    set_target_properties(ort_tensor_subscriber_component PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE)
    target_link_libraries(ort_tensor_subscriber_component
      test_fixture
      CUDA::cudart
      rclcpp::rclcpp
      rclcpp_components::component
      rosidl_runtime_cpp::rosidl_runtime_cpp
      ${std_msgs_TARGETS}
      ${tensor_msgs_TARGETS}
    )
    rclcpp_components_register_node(ort_tensor_subscriber_component
      PLUGIN "OrtTensorSubscriber"
      EXECUTABLE ort_tensor_subscriber_node
    )

    install(
      TARGETS
        ort_tensor_publisher_component
        ort_tensor_subscriber_component
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin
    )

    add_launch_test(test/test_ort_tensor_inter_pubsub_fastrtps_launch.py
      TARGET test_ort_tensor_inter_pubsub_fastrtps_launch
      TIMEOUT 30
    )
  endif()
endif()

ament_package(
  CONFIG_EXTRAS_POST
    cmake/onnxruntime_conversions-extras.cmake
)
