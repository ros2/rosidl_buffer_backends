cmake_minimum_required(VERSION 3.20)
project(python_onnxruntime_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(onnxruntime_core_vendor REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_runtime_version "${ONNXRUNTIME_CORE_VENDOR_VERSION}")
if(NOT _runtime_version)
  set(_runtime_version "${onnxruntime_core_vendor_VERSION}")
endif()

set(_python_package "onnxruntime")
set(_index_url "https://pypi.org/simple")

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheel")
set(_stage_root "${CMAKE_CURRENT_BINARY_DIR}/python_stage")
set(_python_stage "${_stage_root}/${PYTHON_INSTALL_DIR}")
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
string(CONCAT _stage_marker
  "${_stage_root}/.${_python_package}-${_runtime_version}-"
  "cpu-"
  "py${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}-${_system_arch}")

get_filename_component(_installed_capi
  "${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}/onnxruntime/capi"
  ABSOLUTE)
file(RELATIVE_PATH _relative_core_lib
  "${_installed_capi}" "${onnxruntime_core_vendor_LIBRARY_DIRS}")

if(NOT EXISTS "${_stage_marker}")
  file(REMOVE_RECURSE "${_wheel_dir}" "${_stage_root}")
  file(MAKE_DIRECTORY "${_wheel_dir}" "${_python_stage}")

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip download
      --disable-pip-version-check
      --no-cache-dir
      --no-deps
      --only-binary=:all:
      --dest "${_wheel_dir}"
      --index-url "${_index_url}"
      "${_python_package}==${_runtime_version}"
    COMMAND_ERROR_IS_FATAL ANY)

  file(GLOB _onnxruntime_wheels "${_wheel_dir}/*.whl")
  list(LENGTH _onnxruntime_wheels _wheel_count)
  if(NOT _wheel_count EQUAL 1)
    message(FATAL_ERROR
      "${PROJECT_NAME}: expected one Python wheel, found ${_wheel_count}")
  endif()

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip install
      --disable-pip-version-check
      --no-cache-dir
      --no-deps
      --only-binary=:all:
      --target "${_python_stage}"
      "${_onnxruntime_wheels}"
    COMMAND_ERROR_IS_FATAL ANY)

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONNOUSERSITE=1"
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/stage_onnxruntime.py"
      "${_python_stage}"
      "${onnxruntime_core_vendor_LIBRARY_DIRS}"
      "${_relative_core_lib}"
      "${_python_package}"
      "${_runtime_version}"
    COMMAND_ERROR_IS_FATAL ANY)
  file(WRITE "${_stage_marker}" "")
endif()

_ament_cmake_python_register_environment_hook()
install(DIRECTORY "${_python_stage}/"
  DESTINATION "${PYTHON_INSTALL_DIR}"
  USE_SOURCE_PERMISSIONS)
install(FILES LICENSE_ONNXRUNTIME README.md DESTINATION "share/${PROJECT_NAME}")

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_python_onnxruntime_vendor
    test/test_python_onnxruntime_vendor.py
    TIMEOUT 60)
  set_tests_properties(test_python_onnxruntime_vendor PROPERTIES
    ENVIRONMENT
      "PYTHON_ONNXRUNTIME_VENDOR_PACKAGE=${_python_package};\
PYTHON_ONNXRUNTIME_VENDOR_VERSION=${_runtime_version};\
PYTHON_ONNXRUNTIME_VENDOR_CORE_LIBDIR=${onnxruntime_core_vendor_LIBRARY_DIRS}")
endif()

ament_package()
