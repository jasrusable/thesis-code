project(meh)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

add_executable(meh main.cpp)

target_link_libraries(meh ${OpenCV_LIBS})
