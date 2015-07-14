cmake_minimum_required (VERSION 2.6)
project (my_app)

find_package(OpenCV REQUIRED)

# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
	"${PROJECT_SOURCE_DIR}/my_app.h.in"
	"${PROJECT_SOURCE_DIR}/my_app.h"
	)

# add the binary tree to the search path for include files
# so that we will find my_app.h
include_directories("${PROJECT_BINARY_DIR}")

add_executable(my_app my_app.cpp)

target_link_libraries(my_app ${OpenCV_LIBS})
