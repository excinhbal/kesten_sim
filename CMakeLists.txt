cmake_minimum_required(VERSION 3.21)
project(kesten_sim)

set(CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
find_package(MPI REQUIRED COMPONENTS CXX)
add_definitions(-DOMPI_SKIP_MPICXX)

#find_package(Python3 3.7 COMPONENTS Interpreter NumPy REQUIRED)
add_executable(kesten_sim_mpi main_mpi.cpp kestensimulation.cpp mpikesten.cpp)
target_link_libraries(kesten_sim_mpi PRIVATE MPI::MPI_CXX)

add_executable(kesten_sim main.cpp kestensimulation.cpp)

add_executable(test_kesten_sim test_kestensim.cpp kestensimulation.cpp)
#target_link_libraries(kesten_sim Python3::NumPy)
