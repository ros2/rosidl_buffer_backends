cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions_cpu)

find_package(ament_cmake REQUIRED)
find_package(onnxruntime 1.23.2 EXACT CONFIG REQUIRED)
find_package(onnxruntime_conversions REQUIRED COMPONENTS Interface)
find_package(pluginlib REQUIRED)

get_target_property(_ort_library onnxruntime::onnxruntime IMPORTED_LOCATION)
if(NOT _ort_library)
  get_target_property(
    _ort_library onnxruntime::onnxruntime IMPORTED_LOCATION_RELEASE)
endif()
if(_ort_library)
  get_filename_component(_ort_library_dir "${_ort_library}" DIRECTORY)
  if(EXISTS "${_ort_library_dir}/libonnxruntime_providers_cuda.so")
    message(FATAL_ERROR
      "${PROJECT_NAME} requires the system CPU-only ONNX Runtime 1.23.2")
  endif()
endif()

add_library(${PROJECT_NAME}_impl SHARED
  src/onnxruntime_conversions_cpu_impl.cpp)
target_compile_definitions(${PROJECT_NAME}_impl
  PRIVATE ONNXRUNTIME_CONVERSIONS_BUILDING_DLL)
target_compile_features(${PROJECT_NAME}_impl PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_impl
  PUBLIC
    onnxruntime_conversions::interface
    onnxruntime::onnxruntime
  PRIVATE ${CMAKE_DL_LIBS})

add_library(${PROJECT_NAME}_plugin SHARED
  src/cpu_conversion_plugin.cpp)
target_compile_features(${PROJECT_NAME}_plugin PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_plugin
  PRIVATE
    ${PROJECT_NAME}_impl
    onnxruntime::onnxruntime)

pluginlib_export_plugin_description_file(
  onnxruntime_conversions onnxruntime_conversions_cpu.xml)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/onnxruntime_conversions__runtime)
install(
  TARGETS ${PROJECT_NAME}_impl
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(
  TARGETS ${PROJECT_NAME}_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_export_targets(export_${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  HAS_LIBRARY_TARGET)
ament_export_dependencies(onnxruntime)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  ament_add_gtest(test_onnxruntime_conversions_cpu
    test/test_onnxruntime_conversions_cpu.cpp)
  if(TARGET test_onnxruntime_conversions_cpu)
    target_link_libraries(test_onnxruntime_conversions_cpu
      ${PROJECT_NAME}_impl onnxruntime::onnxruntime pluginlib::pluginlib)
    add_dependencies(test_onnxruntime_conversions_cpu ${PROJECT_NAME}_plugin)
  endif()
  ament_add_gtest(test_plugin_catalog test/test_plugin_catalog.cpp)
  if(TARGET test_plugin_catalog)
    target_link_libraries(test_plugin_catalog onnxruntime_conversions::interface)
  endif()
endif()

ament_package()
