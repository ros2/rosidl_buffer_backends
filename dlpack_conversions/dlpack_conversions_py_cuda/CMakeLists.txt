cmake_minimum_required(VERSION 3.20)
project(dlpack_conversions_py_cuda)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(dlpack_conversions_cuda)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION
    share/ament_index/resource_index/dlpack_conversions__python_plugins
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_dlpack_conversions_py_cuda
    test/test_dlpack_conversions_py_cuda.py
    TIMEOUT 60
  )
endif()

ament_package()
