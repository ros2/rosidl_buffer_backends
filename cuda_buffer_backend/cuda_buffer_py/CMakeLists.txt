cmake_minimum_required(VERSION 3.20)
project(cuda_buffer_py)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(cuda_buffer REQUIRED)
find_package(rosidl_buffer REQUIRED)

# Work around Ubuntu's split nvidia-cuda-dev layout in bloom builds.
find_path(_CUDA_RUNTIME_DIR
  NAMES cuda_runtime.h
  HINTS
    ENV CUDA_HOME
    ENV CUDA_PATH
    /usr/local/cuda/include
    /usr/include
)
if(_CUDA_RUNTIME_DIR AND NOT DEFINED CUDAToolkit_INCLUDE_DIRECTORIES)
  set(CUDAToolkit_INCLUDE_DIRECTORIES "${_CUDA_RUNTIME_DIR}" CACHE PATH
    "CUDA include dir (FindCUDAToolkit nvidia-cuda-dev workaround)")
endif()
find_package(CUDAToolkit REQUIRED)

# Find Python before pybind11 so both use the same interpreter.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_cuda_buffer_py
  src/cuda_buffer_py.cpp
)

target_link_libraries(_cuda_buffer_py PRIVATE
  CUDA::cudart
  cuda_buffer::cuda_buffer
  rosidl_buffer::rosidl_buffer
)

# Install the Python package first so ament defines PYTHON_INSTALL_DIR.
ament_python_install_package(cuda_buffer)

install(TARGETS _cuda_buffer_py
  DESTINATION "${PYTHON_INSTALL_DIR}/cuda_buffer"
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_cuda_buffer_py
    test/test_cuda_buffer_py.py
    TIMEOUT 60
  )

  ament_add_pytest_test(test_cuda_buffer_py_pubsub
    test/test_cuda_buffer_py_pubsub.py
    TIMEOUT 40
  )
endif()

ament_package()
