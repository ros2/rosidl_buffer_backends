cmake_minimum_required(VERSION 3.20)
project(cuda_buffer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(cuda_buffer_backend_msgs REQUIRED)
find_package(rmw REQUIRED)
find_package(rcutils REQUIRED)

# CUDA 13 only. Consumers of this backend build against CUDA 13, and a
# cuda_buffer linked against CUDA 12 pulls a second CUDA runtime into every
# process that loads the backend. FindCUDAToolkit treats a requested version as
# a minimum and does not handle version ranges (no HANDLE_VERSION_RANGE on its
# find_package_handle_standard_args call), so the upper bound is checked here.
#
# This requires the NVIDIA cuda-toolkit-13 packages, which install a complete
# toolkit under /usr/local/cuda-13.x. It deliberately does not fall back to
# Ubuntu's split nvidia-cuda-toolkit layout: that ships CUDA 12.x on Noble and
# was what previously got picked up in the bloom build pipeline.
find_package(CUDAToolkit 13 REQUIRED)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 14)
  message(FATAL_ERROR
    "cuda_buffer requires CUDA 13.x; found ${CUDAToolkit_VERSION}")
endif()

add_library(${PROJECT_NAME} SHARED
  src/cuda_buffer.cpp
  src/cuda_buffer_impl.cpp
  src/cuda_buffer_ipc_manager.cpp
  src/cuda_memory_pool.cpp
  src/host_endpoint_manager.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    rosidl_buffer::rosidl_buffer
    rmw::rmw
    rcutils::rcutils
    ${cuda_buffer_backend_msgs_TARGETS}
  PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
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
ament_export_dependencies(rosidl_buffer cuda_buffer_backend_msgs rmw rcutils)
ament_export_libraries(${PROJECT_NAME})
ament_export_include_directories(include/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_cuda_buffer
    test/test_cuda_buffer.cpp
  )
  if(TARGET test_cuda_buffer)
    target_link_libraries(test_cuda_buffer
      ${PROJECT_NAME}
      rosidl_buffer::rosidl_buffer
      CUDA::cudart
      CUDA::cuda_driver
      rt
    )
  endif()

endif()

ament_package()
