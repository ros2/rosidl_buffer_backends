cmake_minimum_required(VERSION 3.20)
project(torch_conversions_py)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(cuda_buffer QUIET)

if(cuda_buffer_FOUND)
  find_package(libtorch_vendor REQUIRED)
  find_package(rosidl_buffer REQUIRED)

  find_path(TORCH_DLPACK_INCLUDE_DIR
    NAMES ATen/dlpack.h
    HINTS
      "${libtorch_vendor_DIR}/../../../opt/libtorch_vendor/include"
    REQUIRED
  )

  # Find Python before pybind11 so both use the same interpreter.
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)

  pybind11_add_module(_torch_conversions_py
    src/torch_conversions_py.cpp
  )

  target_include_directories(_torch_conversions_py PRIVATE
    "${TORCH_DLPACK_INCLUDE_DIR}"
  )

  target_link_libraries(_torch_conversions_py PRIVATE
    cuda_buffer::cuda_buffer
    rosidl_buffer::rosidl_buffer
  )
endif()

# Install the Python package first so ament defines PYTHON_INSTALL_DIR.
ament_python_install_package(torch_conversions)

if(cuda_buffer_FOUND)
  install(TARGETS _torch_conversions_py
    DESTINATION "${PYTHON_INSTALL_DIR}/torch_conversions"
  )
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_torch_conversions_py
    test/test_torch_conversions_py.py
    TIMEOUT 90
  )

  if(cuda_buffer_FOUND)
    install(PROGRAMS test/torch_tensor_publisher_node.py
      DESTINATION "lib/${PROJECT_NAME}"
      RENAME torch_tensor_publisher_node
    )

    add_launch_test(test/test_torch_tensor_inter_pubsub_fastrtps.py
      TARGET test_torch_tensor_inter_pubsub_fastrtps
      TIMEOUT 40
    )
  endif()
endif()

ament_package()
