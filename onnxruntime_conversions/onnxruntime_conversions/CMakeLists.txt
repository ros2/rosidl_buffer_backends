cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(onnxruntime_core_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(tensor_msgs REQUIRED)

add_library(onnxruntime_conversions SHARED
  src/onnxruntime_conversions.cpp
)
target_compile_definitions(onnxruntime_conversions
  PRIVATE ONNXRUNTIME_CONVERSIONS_BUILDING_DLL)
target_compile_features(onnxruntime_conversions PUBLIC cxx_std_20)
target_include_directories(onnxruntime_conversions
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/onnxruntime_conversions>
)
target_link_libraries(onnxruntime_conversions
  PUBLIC
    onnxruntime::onnxruntime
    rosidl_buffer::rosidl_buffer
    tensor_msgs::tensor_msgs__rosidl_generator_cpp
    pluginlib::pluginlib
  PRIVATE
    ${CMAKE_DL_LIBS}
)

add_library(onnxruntime_conversions_cpu_plugin SHARED
  src/cpu_conversion_backend.cpp
)
target_compile_features(onnxruntime_conversions_cpu_plugin PUBLIC cxx_std_20)
target_link_libraries(onnxruntime_conversions_cpu_plugin
  PRIVATE
    onnxruntime_conversions
    onnxruntime::onnxruntime
    pluginlib::pluginlib
)

pluginlib_export_plugin_description_file(
  onnxruntime_conversions
  onnxruntime_conversions_plugins.xml
)

install(
  DIRECTORY include/
  DESTINATION include/onnxruntime_conversions
)
install(
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)
install(
  TARGETS onnxruntime_conversions onnxruntime_conversions_cpu_plugin
  EXPORT onnxruntime_conversions
  INCLUDES DESTINATION include/onnxruntime_conversions
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(onnxruntime_conversions HAS_LIBRARY_TARGET)
ament_export_dependencies(
  onnxruntime_core_vendor
  pluginlib
  rosidl_buffer
  tensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_backend_catalog
    test/test_backend_catalog.cpp
  )
  if(TARGET test_backend_catalog)
    target_include_directories(test_backend_catalog PRIVATE src)
    target_link_libraries(test_backend_catalog
      onnxruntime_conversions
      onnxruntime::onnxruntime
      pluginlib::pluginlib
    )
  endif()

  ament_add_gtest(test_onnxruntime_conversions
    test/test_onnxruntime_conversions.cpp
  )
  if(TARGET test_onnxruntime_conversions)
    target_link_libraries(test_onnxruntime_conversions
      onnxruntime_conversions
      onnxruntime::onnxruntime
      pluginlib::pluginlib
    )
    add_dependencies(
      test_onnxruntime_conversions
      onnxruntime_conversions_cpu_plugin
    )
  endif()
endif()

ament_package()
