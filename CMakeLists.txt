cmake_minimum_required(VERSION 3.0.2)

# Set the name of the project to smalls
project(smalls)

# Setup the variables for CUDA

# CUDA Toolkit 7.0 and up is required as it is the first release officially supporting c++11.
set(CUDA_TOOLKIT_MINIMUM_VERSION "7.0")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
	message(FATAL_ERROR "CUDA could not be found on this platform. Please make sure that CUDA is installed and nvcc is in the path (try nvcc -V from a console).")
endif(NOT CUDA_FOUND)

if("${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" LESS ${CUDA_TOOLKIT_MINIMUM_VERSION})
	message(FATAL_ERROR "The minimum CUDA Toolkit version for the project is ${CUDA_TOOLKIT_MINIMUM_VERSION}. CMake found version ${CUDA_VERSION}.")
endif("${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" LESS ${CUDA_TOOLKIT_MINIMUM_VERSION})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --keep)

add_subdirectory(ext/googletest)

include_directories(${CMAKE_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

add_subdirectory(tst)
