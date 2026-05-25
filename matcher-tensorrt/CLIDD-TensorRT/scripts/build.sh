#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== Building CLIDD-TensorRT ==="

rm -rf build
mkdir -p build

cmake .. \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBTORCH_ROOT=/home/ardyseto/libtorch \
  -DTENSORRT_ROOT=/home/ardyseto/tensorrt

cmake --build build -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: build/test_inference_cpp"
