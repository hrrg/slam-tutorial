# CMakeLists.txt in stereo_calibration directory
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
)

set( THIRD_PARTY_LIBS
    ${OpenCV_LIBS})
# Define the library containing the stereo_calibration sources
add_library(stereo_calibration 
    stereo_calibration.cpp
)

# Specify include directories for the library
target_include_directories(stereo_calibration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optionally, you can link to other libraries if needed
# target_link_libraries(stereo_calibration PUBLIC other_library)
target_link_libraries(stereo_calibration ${THIRD_PARTY_LIBS})
