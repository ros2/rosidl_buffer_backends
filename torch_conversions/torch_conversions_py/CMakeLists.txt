cmake_minimum_required(VERSION 3.20)
project(torch_conversions_py)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_torch_bridge src/torch_bridge.cpp)

ament_python_install_package(torch_conversions)
install(TARGETS _torch_bridge
  DESTINATION "${PYTHON_INSTALL_DIR}/torch_conversions")

install(PROGRAMS examples/torch_tensor_publisher_node.py
  DESTINATION "lib/${PROJECT_NAME}"
  RENAME torch_tensor_publisher_node)

install(PROGRAMS examples/torch_tensor_subscriber_node.py
  DESTINATION "lib/${PROJECT_NAME}"
  RENAME torch_tensor_subscriber_node)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_torch_conversions_py
    test/test_torch_conversions_py.py TIMEOUT 90)
endif()

ament_package()
