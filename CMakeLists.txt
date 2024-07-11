cmake_minimum_required(VERSION 3.26)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DRELEASE")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_RPATH "/Library/Frameworks/")

find_package(SDL2_IMAGE REQUIRED)
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(NeuralNetwork
        Window.cpp
        Window.h
        main_graph_training.cpp
        NeuralNetwork.cpp
        NeuralNetwork.h
)
target_link_libraries(NeuralNetwork ${SDL2_LIBRARIES} SDL2_image::SDL2_image)