cmake_minimum_required(VERSION 2.8.0)

# Set the name of the project to DragonFly
project(smalls)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/googletest/include/)
message("You need to pull googletest here: " ${CMAKE_SOURCE_DIR})

# Setup the variables for CUDA

# CUDA Toolkit 7.0 and up is required as it is the first release officially supporting c++11.
set(CUDA_TOOLKIT_MINIMUM_VERSION "7.0")

find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
	message(FATAL_ERROR "CUDA could not be found on this platform. Please make sure that CUDA is installed and nvcc is in the path (try nvcc -V from a console).")
endif(NOT CUDA_FOUND)

if("${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" LESS ${CUDA_TOOLKIT_MINIMUM_VERSION})
	message(FATAL_ERROR "The minimum CUDA Toolkit version for the project is ${CUDA_TOOLKIT_MINIMUM_VERSION}. CMake found version ${CUDA_VERSION}.")
endif("${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" LESS ${CUDA_TOOLKIT_MINIMUM_VERSION})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --keep)

include_directories(${CUDA_INCLUDE_DIRS})


add_subdirectory(googletest)
add_subdirectory(inc)
add_subdirectory(tst)