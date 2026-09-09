cmake_minimum_required(VERSION 3.20)
project(torch_conversions_cuda)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(torch_conversions REQUIRED COMPONENTS Interface)
find_package(libtorch_vendor REQUIRED)
find_package(Torch 2.9.1 EXACT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(cuda_buffer REQUIRED)

if(NOT TARGET torch_cuda)
  message(FATAL_ERROR
    "${PROJECT_NAME}: libtorch_vendor did not provide CUDA-enabled LibTorch")
endif()

add_library(torch_conversions_cuda_impl SHARED
  src/torch_conversions_cuda_impl.cpp
)
target_compile_definitions(torch_conversions_cuda_impl PRIVATE
  TORCH_CONVERSIONS_BUILDING_DLL
  __NV_NO_HOST_COMPILER_CHECK=1
)
target_link_libraries(torch_conversions_cuda_impl
  PUBLIC torch_conversions::interface torch
  PRIVATE CUDA::cudart pluginlib::pluginlib
)

add_library(torch_conversions_cuda_plugin SHARED
  src/cuda_conversion_plugin.cpp
)
target_compile_definitions(torch_conversions_cuda_plugin PRIVATE
  __NV_NO_HOST_COMPILER_CHECK=1
)
target_link_libraries(torch_conversions_cuda_plugin PRIVATE
  torch_conversions::interface
  CUDA::cudart
  cuda_buffer::cuda_buffer
  pluginlib::pluginlib
)

pluginlib_export_plugin_description_file(
  torch_conversions
  torch_conversions_cuda.xml
)

install(
  TARGETS torch_conversions_cuda_impl torch_conversions_cuda_plugin
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
ament_export_dependencies(
  libtorch_vendor
  Torch
  CUDAToolkit
  cuda_buffer
  pluginlib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_torch_conversions_cuda
    test/test_torch_conversions_cuda.cpp
  )
  if(TARGET test_torch_conversions_cuda)
    target_link_libraries(test_torch_conversions_cuda
      torch_conversions_cuda_impl
      torch
      CUDA::cuda_driver
    )
    add_dependencies(
      test_torch_conversions_cuda
      torch_conversions_cuda_plugin
    )
  endif()
endif()

ament_package()
