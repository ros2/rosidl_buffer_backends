cmake_minimum_required(VERSION 3.20)
project(torch_conversions_py)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_dlpack_bridge src/dlpack_bridge.cpp)

ament_python_install_package(torch_conversions)
install(TARGETS _dlpack_bridge
  DESTINATION "${PYTHON_INSTALL_DIR}/torch_conversions"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
