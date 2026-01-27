#!/bin/bash
# Download ONNX Runtime for C++ ONNX matcher implementations
# This script downloads the ONNX Runtime GPU package with CUDA support

set -e

ONNX_VERSION="1.20.1"
CUDA_VERSION="12.6"  # ONNX Runtime 1.20.1 uses CUDA 12.6
PLATFORM="linux-x64"

ONNX_DIR="matcher-cpp-onnx/onnxruntime"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-${PLATFORM}-gpu-${ONNX_VERSION}.tgz"

echo "======================================================="
echo "📦 Downloading ONNX Runtime v${ONNX_VERSION}"
echo "======================================================="
echo "Platform:     ${PLATFORM}"
echo "CUDA Support: Yes (CUDA ${CUDA_VERSION})"
echo "Download URL: ${DOWNLOAD_URL}"
echo "Target Dir:   ${ONNX_DIR}"
echo "======================================================="
echo

# Check if directory already exists
if [ -d "$ONNX_DIR" ]; then
    echo "⚠️  ONNX Runtime directory already exists: $ONNX_DIR"
    read -p "Do you want to remove it and re-download? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing directory..."
        rm -rf "$ONNX_DIR"
    else
        echo "ℹ️  Keeping existing directory. Exiting."
        exit 0
    fi
fi

# Create temporary directory
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo "📥 Downloading ONNX Runtime..."
cd "$TMP_DIR"
wget -q --show-progress "$DOWNLOAD_URL" -O onnxruntime.tgz

if [ $? -ne 0 ]; then
    echo "❌ Download failed!"
    exit 1
fi

echo ""
echo "📦 Extracting archive..."
tar -xzf onnxruntime.tgz

# Find the extracted directory (it will be named onnxruntime-linux-x64-gpu-1.20.1)
EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "onnxruntime-*" | head -n 1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "❌ Extraction failed! Could not find extracted directory."
    exit 1
fi

echo "✅ Archive extracted successfully"
echo ""
echo "📂 Moving to target directory..."

cd -
mkdir -p "matcher-cpp-onnx"
mv "$TMP_DIR/$EXTRACTED_DIR" "$ONNX_DIR"

echo "✅ ONNX Runtime installed successfully!"
echo ""
echo "======================================================="
echo "📋 Summary"
echo "======================================================="
echo "Version:      ${ONNX_VERSION}"
echo "Location:     ${ONNX_DIR}"
echo "Size:         $(du -sh $ONNX_DIR | cut -f1)"
echo "======================================================="
echo ""
echo "Library files:"
ls -lh "$ONNX_DIR/lib/"*.so* | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "✅ Ready to build C++ ONNX matchers!"
echo ""
echo "To verify:"
echo "  ls -lh $ONNX_DIR/include"
echo "  ls -lh $ONNX_DIR/lib"
echo ""
