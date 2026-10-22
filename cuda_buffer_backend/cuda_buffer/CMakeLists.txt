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

# Workaround for CMake's FindCUDAToolkit on Ubuntu's split nvidia-cuda-dev
# layout in the bloom build pipeline. Pre-populate the cache variable to
# skip the broken auto-detection.
# Related: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=995122
find_path(_CUDA_RUNTIME_DIR
  NAMES cuda_runtime.h
  HINTS
    ENV CUDA_HOME
    ENV CUDA_PATH
    /usr/local/cuda/include
    /usr/include
)
if(_CUDA_RUNTIME_DIR AND NOT DEFINED CUDAToolkit_INCLUDE_DIRECTORIES)
  set(CUDAToolkit_INCLUDE_DIRECTORIES "${_CUDA_RUNTIME_DIR}" CACHE PATH
    "CUDA include dir (FindCUDAToolkit nvidia-cuda-dev workaround)")
endif()
find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/cuda_buffer.cpp
  src/cuda_buffer_impl.cpp
  src/cuda_buffer_ipc_manager.cpp
  src/cuda_memory_pool.cpp
  src/external_memory_pool.cpp
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

  # A stand-in RMW for test_transport_pool. Built -fvisibility=hidden like a
  # real one, so the explicit default visibility on the protocol's two entry
  # points is genuinely exercised rather than incidentally true.
  add_library(rmw_faketransport MODULE test/fake_transport_provider.cpp)
  target_link_libraries(rmw_faketransport ${PROJECT_NAME})
  set_target_properties(rmw_faketransport PROPERTIES
    PREFIX "lib"
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden)

  ament_add_gtest(test_transport_pool
    test/test_transport_pool.cpp
  )
  if(TARGET test_transport_pool)
    target_link_libraries(test_transport_pool
      ${PROJECT_NAME}
      rosidl_buffer::rosidl_buffer
      CUDA::cudart
      CUDA::cuda_driver
      rt
    )
    add_dependencies(test_transport_pool rmw_faketransport)
    # The client resolves the provider by soname, so it has to be findable the
    # way rmw_implementation would find a real RMW.
    set_property(TEST test_transport_pool APPEND PROPERTY
      ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rmw_faketransport>:$ENV{LD_LIBRARY_PATH}")
  endif()

  ament_add_gtest(test_external_memory_pool
    test/test_external_memory_pool.cpp
  )
  if(TARGET test_external_memory_pool)
    target_link_libraries(test_external_memory_pool
      ${PROJECT_NAME}
      rosidl_buffer::rosidl_buffer
      CUDA::cudart
      CUDA::cuda_driver
      rt
    )
  endif()

endif()

ament_package()
