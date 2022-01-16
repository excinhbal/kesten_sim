cmake_minimum_required(VERSION 3.21)
project(kesten_sim)

set(CMAKE_CXX_STANDARD 20)

#find_package(Python3 3.7 COMPONENTS Interpreter NumPy REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
add_definitions(-DOMPI_SKIP_MPICXX)
add_executable(kesten_sim_mpi main_mpi.cpp)
target_link_libraries(kesten_sim_mpi PRIVATE MPI::MPI_CXX)

set (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
add_executable(kesten_sim main.cpp)
add_executable(test_kesten_sim test_kestensim.cpp)
#target_link_libraries(kesten_sim Python3::NumPy)
