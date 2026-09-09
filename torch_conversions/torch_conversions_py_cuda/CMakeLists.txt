cmake_minimum_required(VERSION 3.20)
project(torch_conversions_py_cuda)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(torch_conversions_cuda)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION
    share/ament_index/resource_index/torch_conversions__python_plugins
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
