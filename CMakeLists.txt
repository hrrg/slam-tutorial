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
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tsl-robin-map REQUIRED)
find_package(Sophus REQUIRED)
find_package(TBB REQUIRED)
set(CMAKE_BUILD_TYPE Release)
# add_library(stereo_calibration "")

# target_sources(stereo_calibration 
# PRIVATE ${CMAKE_CURRENT_LIST_DIR}/executables/stereo_calibration.cpp
# PUBLIC ${CMAKE_CURRENT_LIST_DIR}/executables/stereo_calibration.h)
# target_link_libraries(stereo_calibration  ${THIRD_PARTY_LIBS} )

# target_include_directories(stereo_calibration
# PUBLIC ${CMAKE_CURRENT_LIST_DIR}/executables)

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

add_subdirectory(src)

function(custom_executable target)
  add_executable(${target}_node executables/${target}.cpp)
  include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/eigen3
    /usr/include/vtk-5.8
  )
  target_link_libraries(${target}_node 
  stereo_calibration
  TBB::tbb 
  )
  ament_target_dependencies(${target}_node
    "rclcpp"
    "sensor_msgs"
    "image_transport"
    "OpenCV"
    "Eigen3"
    "tf2_ros"
    "nav_msgs"
    "TBB"
    )
  install(TARGETS ${target}_node
  DESTINATION lib/${PROJECT_NAME})
endfunction()

# custom_executable(main)
custom_executable(main_lidar)

ament_package()


install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)
