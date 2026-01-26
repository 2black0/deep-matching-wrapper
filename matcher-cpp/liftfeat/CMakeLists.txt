cmake_minimum_required(VERSION 3.16)

project(dmw_liftfeat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Silence a harmless PyTorch CMake warning:
# "Failed to compute shorthash for libnvrtc.so".
# PyTorch falls back to "XXXXXXXX" internally anyway.
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

add_library(dmw_liftfeat_lib
  src/LiftFeatTorchMatcher.cpp
)

target_include_directories(dmw_liftfeat_lib
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dmw_liftfeat_lib
  PUBLIC
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

add_executable(demo_liftfeat demo/demo_liftfeat.cpp)

target_link_libraries(demo_liftfeat PRIVATE dmw_liftfeat_lib)
