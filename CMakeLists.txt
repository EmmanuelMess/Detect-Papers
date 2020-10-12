cmake_minimum_required(VERSION 3.17)
project(opencvtest)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_STATIC ON)

find_package( OpenCV 4.1.2 REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencvtest main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

target_link_libraries(opencvtest ${OpenCV_LIBS})
