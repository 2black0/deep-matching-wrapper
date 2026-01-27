cmake_minimum_required(VERSION 3.16)

project(dmw_lightglue LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use pkg-config to avoid OpenCV's CUDA version requirement
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
set(OpenCV_LIBS ${OPENCV_LIBRARIES})

# Silence a harmless PyTorch CMake warning
if(NOT DEFINED CUDA_NVRTC_SHORTHASH)
  set(CUDA_NVRTC_SHORTHASH "XXXXXXXX" CACHE STRING "NVRTC shorthash override" FORCE)
endif()

# LibTorch
set(LIBTORCH_DIR "/home/ardyseto/libtorch" CACHE PATH "LibTorch root directory")

if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
  if(EXISTS "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
  endif()
endif()

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(dmw_lightglue_lib
  src/SuperPointLightGlueMatcher.cpp
)

target_include_directories(dmw_lightglue_lib
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dmw_lightglue_lib
  PUBLIC
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

add_executable(demo_superpoint_lightglue demo/demo_superpoint_lightglue.cpp)
target_link_libraries(demo_superpoint_lightglue PRIVATE dmw_lightglue_lib)
