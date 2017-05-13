cmake_minimum_required(VERSION 2.8.7)

# Set the name of the project to smalls
project(smalls)

# Setup the variables for CUDA

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

add_subdirectory(ext/googletest)
include_directories(${CMAKE_SOURCE_DIR} ${GTEST_INCLUDE_DIRS})
add_subdirectory(tst/tst_host)

option(WITH_CUDA "Build with CUDA" OFF)
# CUDA Toolkit 7.0 and up is required as it is the first release officially supporting c++11.

if(WITH_CUDA)
    find_package(CUDA 7.0 REQUIRED)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --keep)
    add_subdirectory(tst/tst_cuda)
endif(WITH_CUDA)

