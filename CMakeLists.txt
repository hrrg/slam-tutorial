cmake_minimum_required(VERSION 3.5)
project(slam-tutorial)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


message(${PCL_INCLUDE_DIRS})
function(custom_executable target)
  add_executable(${target}_node executables/${target}.cpp)
  include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    /usr/include/vtk-5.8
  )
  target_link_libraries(${target}_node ${PCL_LIBRARIES} )
  ament_target_dependencies(${target}_node
    "rclcpp"
    "sensor_msgs"
    "image_transport"
    "OpenCV"
    "PCL"
    )
  install(TARGETS ${target}_node
  DESTINATION lib/${PROJECT_NAME})
endfunction()

custom_executable(main)
custom_executable(lidar_subscriber)

ament_package()


install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)
