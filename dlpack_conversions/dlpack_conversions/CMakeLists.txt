cmake_minimum_required(VERSION 3.20)
project(dlpack_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rcutils REQUIRED)
find_package(rmw REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(tensor_msgs REQUIRED)

# Headers and message dependencies shared with the storage plugins, which
# implement the interface without linking the core library.
add_library(interface INTERFACE)
add_library(${PROJECT_NAME}::interface ALIAS interface)
target_compile_features(interface INTERFACE cxx_std_17)
target_include_directories(interface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(interface INTERFACE
  rcutils::rcutils
  rmw::rmw
  rosidl_buffer::rosidl_buffer
  tensor_msgs::tensor_msgs__rosidl_generator_cpp
)

add_library(${PROJECT_NAME} SHARED src/dlpack_conversions.cpp)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME}
  PRIVATE DLPACK_CONVERSIONS_BUILDING_DLL)
target_link_libraries(${PROJECT_NAME}
  PUBLIC interface
  PRIVATE pluginlib::pluginlib ${CMAKE_DL_LIBS}
)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(
  TARGETS interface ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} NAMESPACE ${PROJECT_NAME}::)
ament_export_dependencies(pluginlib rcutils rmw rosidl_buffer tensor_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # dlpack.h is vendored verbatim from the DLPack project.
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_cpplint
    ament_cmake_uncrustify
  )
  ament_lint_auto_find_test_dependencies()
  ament_copyright(EXCLUDE include/dlpack_conversions/dlpack.h)
  ament_cpplint(EXCLUDE include/dlpack_conversions/dlpack.h)
  ament_uncrustify(EXCLUDE include/dlpack_conversions/dlpack.h)
endif()

ament_package()
