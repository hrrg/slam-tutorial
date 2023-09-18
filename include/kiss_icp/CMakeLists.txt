# CMakeLists.txt in stereo_calibration directory
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
)

# Define the library containing the stereo_calibration sources
add_library(kiss_icp 
    kiss_icp.cpp
)

# Specify include directories for the library
target_include_directories(kiss_icp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optionally, you can link to other libraries if needed
# target_link_libraries(stereo_calibration PUBLIC other_library)
target_link_libraries(kiss_icp ${THIRD_PARTY_LIBS})
