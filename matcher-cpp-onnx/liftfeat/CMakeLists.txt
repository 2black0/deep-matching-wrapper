cmake_minimum_required(VERSION 3.16)

project(dmw_liftfeat_onnx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add RPATH for runtime library search
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Find OpenCV
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
set(OpenCV_LIBS ${OPENCV_LIBRARIES})

# Find ONNX Runtime
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime" CACHE PATH "ONNX Runtime root directory")

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_DIR}")
endif()

set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Library
add_library(dmw_liftfeat_onnx_lib
    src/LiftFeatOnnxMatcher.cpp
)

target_include_directories(dmw_liftfeat_onnx_lib
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dmw_liftfeat_onnx_lib
    PUBLIC
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)

# Demo executable
add_executable(liftfeat_demo demo/liftfeat_demo.cpp)

target_link_libraries(liftfeat_demo PRIVATE dmw_liftfeat_onnx_lib)

# Fix TBB ABI issue
target_link_options(liftfeat_demo PRIVATE "-Wl,--allow-shlib-undefined")
