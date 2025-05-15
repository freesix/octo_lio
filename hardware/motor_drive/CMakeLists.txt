cmake_minimum_required(VERSION 3.8)
project(motor_drive)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

include_directories("include/motor_drive")
ament_auto_add_library(${PROJECT_NAME} src/motor_drive.cpp src/crc.cpp)
ament_auto_add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  motor_drive_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


ament_auto_package()
