cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions_py)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_ort_bridge src/ort_bridge.cpp)

ament_python_install_package(onnxruntime_conversions)

install(TARGETS _ort_bridge
  DESTINATION "${PYTHON_INSTALL_DIR}/onnxruntime_conversions")

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_onnxruntime_conversions_py
    test/test_onnxruntime_conversions_py.py TIMEOUT 120)
  ament_add_pytest_test(test_onnxruntime_conversions_py_cuda
    test/test_onnxruntime_conversions_py_cuda.py TIMEOUT 120)
  ament_add_pytest_test(test_onnxruntime_tensor_inter_pubsub_fastrtps
    test/test_onnxruntime_tensor_inter_pubsub_fastrtps.py TIMEOUT 120)
endif()

ament_package()
