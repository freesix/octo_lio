cmake_minimum_required(VERSION 3.8)
project(point_to_map)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_library(point2scan SHARED src/point2scan.cpp)
rclcpp_components_register_node(point2scan
  PLUGIN "point2scan"
  EXECUTABLE point2scan_node)

ament_auto_add_library(gridmap SHARED 
  src/gridmap.cpp 
  src/value_conversion_tables.cpp
  src/probability_values.cpp
  src/grid_2d.cpp
  src/probability_grid.cpp
  src/probability_grid_range_data_inserter.cpp)

rclcpp_components_register_node(gridmap
  PLUGIN "gridmap"
  EXECUTABLE gridmap_node)

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
