cmake_minimum_required(VERSION 3.5.1)
project(kesten_sim)

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++1z")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -std=c++1z")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -ffast-math -std=c++1z")
find_package(MPI REQUIRED COMPONENTS CXX)
add_definitions(-DOMPI_SKIP_MPICXX)

# Kesten
add_executable(kesten_sim_mpi main_mpi.cpp kestensimulation.cpp mpikesten.cpp)
target_link_libraries(kesten_sim_mpi PRIVATE ${MPI_CXX_LIBRARIES})
target_include_directories(kesten_sim_mpi PRIVATE ${MPI_CXX_INCLUDE_PATH})

add_executable(kesten_sim main.cpp kestensimulation.cpp)

add_executable(test_kesten_sim test_kestensim.cpp kestensimulation.cpp)

# Quadratic Kesten
add_executable(quad_kesten_sim_mpi main_mpi_quad.cpp kestensimulation.cpp mpikesten.cpp)
target_link_libraries(quad_kesten_sim_mpi PRIVATE ${MPI_CXX_LIBRARIES})
target_include_directories(quad_kesten_sim_mpi PRIVATE ${MPI_CXX_INCLUDE_PATH})

