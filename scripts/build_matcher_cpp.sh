#!/bin/bash
# Build script for individual matcher-cpp modules
# Usage: bash scripts/build_matcher_cpp.sh <matcher>

set -e

MATCHER=$1

if [ -z "$MATCHER" ]; then
    echo "Usage: $0 <matcher>"
    echo "Available matchers: xfeat, liftfeat, clidd"
    exit 1
fi

# Configuration
export LIBTORCH_DIR="/home/ardyseto/libtorch"
export CUDA_PATH="/usr/local/cuda-13.0"
export OPENCV_LIB_PATH="/usr/local/lib"
export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:$CUDA_PATH/lib64:$OPENCV_LIB_PATH:$LD_LIBRARY_PATH"

# Verify paths
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "❌ Error: LibTorch not found at $LIBTORCH_DIR"
    echo "   Please download from: https://pytorch.org/get-started/locally/"
    exit 1
fi

if [ ! -d "$CUDA_PATH" ]; then
    echo "❌ Error: CUDA 13.0 not found at $CUDA_PATH"
    exit 1
fi

MATCHER_DIR="matcher-cpp/$MATCHER"
BUILD_DIR="$MATCHER_DIR/build"

if [ ! -d "$MATCHER_DIR" ]; then
    echo "❌ Error: Matcher directory not found: $MATCHER_DIR"
    exit 1
fi

echo "======================================================="
echo "🔨 Building matcher-cpp: $MATCHER"
echo "======================================================="
echo "LibTorch: $LIBTORCH_DIR"
echo "CUDA:     $CUDA_PATH"
echo "Build:    $BUILD_DIR"
echo "======================================================="

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure CMake
echo "⚙️  Configuring CMake..."
cmake -S "$MATCHER_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBTORCH_DIR="$LIBTORCH_DIR" \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
    -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
    -DCaffe2_DIR="$LIBTORCH_DIR/share/cmake/Caffe2" \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$OPENCV_LIB_PATH" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$OPENCV_LIB_PATH"

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed!"
    exit 1
fi

# Build
echo "🔨 Building..."
cmake --build "$BUILD_DIR" -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""
echo "Demo binary: ./$BUILD_DIR/demo_$MATCHER"
echo ""
echo "To run:"
echo "  export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:$CUDA_PATH/lib64:$OPENCV_LIB_PATH:\$LD_LIBRARY_PATH"
echo "  ./$BUILD_DIR/demo_$MATCHER --img1 assets/ref.png --img2 assets/tgt.png --device cuda"
