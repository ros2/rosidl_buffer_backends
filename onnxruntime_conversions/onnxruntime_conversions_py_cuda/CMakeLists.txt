cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions_py_cuda)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_onnxruntime_conversions_cuda
    test/test_onnxruntime_conversions_cuda.py
    TIMEOUT 120
  )
  ament_add_pytest_test(test_onnxruntime_tensor_inter_pubsub_fastrtps
    test/test_onnxruntime_tensor_inter_pubsub_fastrtps.py
    TIMEOUT 40
  )
endif()

ament_package()
