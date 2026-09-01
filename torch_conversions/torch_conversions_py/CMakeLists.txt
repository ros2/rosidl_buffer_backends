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
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_dlpack_bridge
  src/dlpack_bridge.cpp
)

ament_python_install_package(torch_conversions)

install(TARGETS _dlpack_bridge
  DESTINATION "${PYTHON_INSTALL_DIR}/torch_conversions"
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_torch_conversions_py
    test/test_torch_conversions_py.py
    TIMEOUT 90
  )

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
