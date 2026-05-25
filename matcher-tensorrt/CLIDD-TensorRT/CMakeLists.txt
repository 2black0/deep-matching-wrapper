cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0146 OLD)
project(clidd_trt_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

find_package(CUDA REQUIRED)

set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
set(OpenCV_LIBS "/usr/local/lib/libopencv_core.so;/usr/local/lib/libopencv_imgproc.so;/usr/local/lib/libopencv_imgcodecs.so;/usr/local/lib/libopencv_calib3d.so")

# Force use SYSTEM LibTorch - bypass pixi completely
set(LIBTORCH_ROOT "/home/ardyseto/libtorch" CACHE PATH "LibTorch installation directory")
set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch")
find_package(Torch REQUIRED)

# Find Python3 for LibTorch (required since LibTorch was built with Python support)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(PYTHON_LIBRARY "/usr/lib/x86_64-linux-gnu/libpython3.10.so")

# TensorRT paths
set(TENSORRT_ROOT "/home/ardyseto/tensorrt" CACHE PATH "TensorRT installation directory")
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# CUDA paths
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(CUDA_LIBRARY_DIR "/usr/local/cuda/lib64")

# OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Compiler flags - suppress warnings from third-party headers (TensorRT)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 ${TORCH_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-parameter")

# Suppress specific TensorRT deprecation warnings via system includes
add_compile_definitions(TRT_LEGACY_PIPELINE=1)

add_library(clidd_trt_lib STATIC
    src/clidd_trt.cpp
)

target_link_libraries(clidd_trt_lib
    PRIVATE
    nvinfer
    cudart
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${PYTHON_LIBRARY}
)

target_include_directories(clidd_trt_lib
    PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
)

set_property(TARGET clidd_trt_lib PROPERTY BUILD_RPATH "${LIBTORCH_ROOT}/lib")

add_executable(test_inference_cpp
    test/test_inference.cpp
)

target_link_libraries(test_inference_cpp
    nvinfer
    cudart
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${PYTHON_LIBRARY}
)

target_include_directories(test_inference_cpp
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
)

set_property(TARGET test_inference_cpp PROPERTY BUILD_RPATH "${LIBTORCH_ROOT}/lib")
