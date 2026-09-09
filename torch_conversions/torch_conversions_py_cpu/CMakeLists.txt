cmake_minimum_required(VERSION 3.20)
project(torch_conversions_py_cpu)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(torch_conversions_cpu)
install(
  FILES resource/${PROJECT_NAME}
  DESTINATION
    share/ament_index/resource_index/torch_conversions__python_plugins
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(test_torch_conversions_py_cpu
    test/test_torch_conversions_py_cpu.py
    TIMEOUT 90
  )
endif()

ament_package()
