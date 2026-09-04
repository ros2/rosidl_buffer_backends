cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions_py)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_onnxruntime_conversions_py
    test/test_onnxruntime_conversions_py.py
    TIMEOUT 60
  )
endif()

ament_package()
