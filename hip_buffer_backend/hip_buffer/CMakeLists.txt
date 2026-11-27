# Derived from the upstream cuda_buffer package; modified for HIP by ObeliskGate, 2026.
cmake_minimum_required(VERSION 3.20)
project(hip_buffer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(hip_buffer_backend_msgs REQUIRED)
find_package(rmw REQUIRED)
find_package(rcutils REQUIRED)

find_package(hip CONFIG REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/hip_buffer.cpp
  src/hip_buffer_impl.cpp
  src/hip_buffer_ipc_manager.cpp
  src/hip_memory_pool.cpp
  src/host_endpoint_manager.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    rosidl_buffer::rosidl_buffer
    rmw::rmw
    rcutils::rcutils
    ${hip_buffer_backend_msgs_TARGETS}
    hip::host
  PRIVATE
    rt
)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rosidl_buffer hip_buffer_backend_msgs rmw rcutils hip)
ament_export_libraries(${PROJECT_NAME})
ament_export_include_directories(include/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_hip_buffer
    test/test_hip_buffer.cpp
  )
  if(TARGET test_hip_buffer)
    target_link_libraries(test_hip_buffer
      ${PROJECT_NAME}
      rosidl_buffer::rosidl_buffer
      rt
    )
  endif()

  ament_add_gtest(test_hip_pool_config
    test/test_hip_pool_config.cpp
  )
  if(TARGET test_hip_pool_config)
    target_link_libraries(test_hip_pool_config
      ${PROJECT_NAME}
      rosidl_buffer::rosidl_buffer
      rt
    )
  endif()

endif()

ament_package()
