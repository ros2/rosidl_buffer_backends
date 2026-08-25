cmake_minimum_required(VERSION 3.20)
project(torch_conversions_py_cuda_plugin)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(cuda_buffer REQUIRED)
find_package(libtorch_vendor REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(torch_conversions_py REQUIRED)

find_path(TORCH_DLPACK_INCLUDE_DIR
  NAMES ATen/dlpack.h
  HINTS
    "${libtorch_vendor_DIR}/../../../opt/libtorch_vendor/include"
  REQUIRED
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_cuda_dlpack_bridge
  src/cuda_dlpack_bridge.cpp
)

target_include_directories(_cuda_dlpack_bridge PRIVATE
  "${TORCH_DLPACK_INCLUDE_DIR}"
)

target_link_libraries(_cuda_dlpack_bridge PRIVATE
  cuda_buffer::cuda_buffer
  rosidl_buffer::rosidl_buffer
)

ament_python_install_package(torch_conversions_cuda_plugin)

install(TARGETS _cuda_dlpack_bridge
  DESTINATION "${PYTHON_INSTALL_DIR}/torch_conversions_cuda_plugin"
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_torch_conversions_cuda
    test/test_torch_conversions_cuda.py
    TIMEOUT 90
  )

  install(PROGRAMS test/torch_tensor_publisher_node.py
    DESTINATION "lib/${PROJECT_NAME}"
    RENAME torch_tensor_publisher_node
  )

  add_launch_test(test/test_torch_tensor_inter_pubsub_fastrtps.py
    TARGET test_torch_tensor_inter_pubsub_fastrtps
    TIMEOUT 40
  )
endif()

ament_package()
