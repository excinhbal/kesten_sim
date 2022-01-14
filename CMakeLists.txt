cmake_minimum_required(VERSION 3.21)
project(kesten_sim)

set(CMAKE_CXX_STANDARD 20)

#find_package(Python3 3.7 COMPONENTS Interpreter NumPy REQUIRED)

set (CMAKE_CXX_FLAGS "-O3 -march=native -ffast-math")
add_executable(kesten_sim main.cpp)
#target_link_libraries(kesten_sim Python3::NumPy)
