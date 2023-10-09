cmake_minimum_required(VERSION 3.18)

project(cuNTT LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(GMP     REQUIRED)

option(BUILD_SHARED "Build as a shared library" OFF)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CUDA_ARCHITECTURES 60)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_compile_options(-O3)

if (BUILD_SHARED)
  add_library(cuNTT SHARED src/cuNTT.cu src/cuSHA.cu)
else()
  add_library(cuNTT STATIC src/cuNTT.cu src/cuSHA.cu)
endif()
target_link_libraries(cuNTT gmp gmpxx cudart)


install(TARGETS cuNTT 
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# Install header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include)
