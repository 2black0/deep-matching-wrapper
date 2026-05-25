#!/bin/bash

# check.sh - Script untuk mengecek dependency DeepV-SLAM Project
# Author: Gemini (untuk Ardy Seto Priambodo)

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================================${NC}"
echo -e "${BLUE}   DeepV-SLAM Environment Checker (Ubuntu)   ${NC}"
echo -e "${BLUE}=============================================================${NC}"

# Fungsi helper untuk print status
# Usage: check_status "NAME" "VERSION" "PATH" "STATUS" ["EXTRA_INFO"]
# EXTRA_INFO format: "Label1: Value1|Label2: Value2|..." (multiple lines separated by |)
check_status() {
    NAME=$1
    VERSION=$2
    PATH_LOC=$3
    STATUS=$4
    EXTRA_INFO=$5

    if [ "$STATUS" == "OK" ]; then
        echo -e "${GREEN}[OK] $NAME${NC}"
        echo -e "     Version : $VERSION"
        echo -e "     Path    : $PATH_LOC"
    elif [ "$STATUS" == "WARNING" ]; then
        echo -e "${YELLOW}[WARNING] $NAME${NC}"
        echo -e "     Version : $VERSION"
        echo -e "     Path    : $PATH_LOC"
    else
        echo -e "${RED}[MISSING] $NAME${NC}"
        echo -e "     Status  : Not found or not in PATH"
    fi

    # Print extra info lines if provided
    if [ ! -z "$EXTRA_INFO" ]; then
        echo "$EXTRA_INFO" | while IFS='|' read -ra LINES; do
            for LINE in "${LINES[@]}"; do
                if [ ! -z "$LINE" ]; then
                    # Parse label and value (format: "Label: Value" or just "Value")
                    if [[ "$LINE" == *": "* ]]; then
                        LABEL=$(echo "$LINE" | cut -d':' -f1)
                        VALUE=$(echo "$LINE" | cut -d':' -f2- | sed 's/^ *//')  # Trim leading space
                        # Calculate padding for alignment (same as "Version", "Path")
                        printf "     %-8s: %s\n" "$LABEL" "$VALUE"
                    else
                        echo -e "     $LINE"
                    fi
                fi
            done
        done
    fi

    echo "-------------------------------------------------------------"
}

# 1. Cek OS
echo -e "Checking Operating System..."
if [ -f /etc/os-release ]; then
    OS_NAME=$(grep -oP '(?<=^PRETTY_NAME=).+' /etc/os-release | tr -d '"')
    echo -e "${GREEN}[OK] OS Found${NC}"
    echo -e "     System  : $OS_NAME"
    echo -e "     Kernel  : $(uname -r)"
else
    echo -e "${RED}[ERROR] Cannot detect OS version${NC}"
fi
echo "-------------------------------------------------------------"

# 2. Cek CMake
CMAKE_PATH=$(which cmake)
if [ ! -z "$CMAKE_PATH" ]; then
    CMAKE_VER=$(cmake --version | head -n1 | awk '{print $3}')
    check_status "CMake" "$CMAKE_VER" "$CMAKE_PATH" "OK"
else
    check_status "CMake" "-" "-" "MISSING"
fi

# 3. Cek GCC
GCC_PATH=$(which gcc)
if [ ! -z "$GCC_PATH" ]; then
    GCC_VER=$(gcc --version | head -n1 | awk '{print $NF}') # Usually last field
    # Alternatif path dari VSCode config user (/usr/lib64/ccache/gcc)
    if [ -f "/usr/lib64/ccache/gcc" ]; then
        echo -e "     Note    : CCache GCC wrapper detected at /usr/lib64/ccache/gcc"
    fi
    check_status "GCC" "$GCC_VER" "$GCC_PATH" "OK"
else
    check_status "GCC" "-" "-" "MISSING"
fi

# 4. Cek CCache
CCACHE_PATH=$(which ccache)
if [ ! -z "$CCACHE_PATH" ]; then
    CCACHE_VER=$(ccache --version | head -n1 | awk '{print $3}')
    check_status "CCache" "$CCACHE_VER" "$CCACHE_PATH" "OK"
else
    check_status "CCache" "-" "-" "MISSING"
fi

# 5. Cek Ninja
NINJA_PATH=$(which ninja)
if [ ! -z "$NINJA_PATH" ]; then
    NINJA_VER=$(ninja --version)
    check_status "Ninja Build" "$NINJA_VER" "$NINJA_PATH" "OK"
else
    check_status "Ninja Build" "-" "-" "MISSING"
fi

# 6. Cek CUDA Toolkit
# Coba via nvcc di PATH
NVCC_PATH=$(which nvcc)
CUDA_VER=""
CUDA_STATUS="MISSING"

if [ ! -z "$NVCC_PATH" ]; then
    CUDA_VER=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
    CUDA_STATUS="OK"
elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
    NVCC_PATH="/usr/local/cuda/bin/nvcc"
    CUDA_VER=$($NVCC_PATH --version | grep release | awk '{print $6}' | cut -c2-)
    CUDA_STATUS="OK"
    CUDA_NOTE="Note: CUDA found but not in system PATH"
fi

check_status "CUDA Toolkit (NVCC)" "$CUDA_VER" "$NVCC_PATH" "$CUDA_STATUS" "$CUDA_NOTE"

# 7. Cek OpenCV
# Coba via pkg-config (metode terbaik untuk C++)
OPENCV_VER=$(pkg-config --modversion opencv4 2>/dev/null)
OPENCV_PATH=$(pkg-config --variable=prefix opencv4 2>/dev/null)

if [ ! -z "$OPENCV_VER" ]; then
    # Cek dukungan CUDA di OpenCV
    OPENCV_LIBS=$(pkg-config --libs opencv4 2>/dev/null)
    OPENCV_CUDA="No"
    if echo "$OPENCV_LIBS" | grep -q "opencv_cuda"; then
        OPENCV_CUDA="Yes (CUDA enabled)"
    fi

    check_status "OpenCV (via pkg-config)" "$OPENCV_VER" "$OPENCV_PATH" "OK" "CUDA: $OPENCV_CUDA"
else
    # Fallback cek python jika pkg-config gagal
    PY_CV_VER=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null)
    if [ ! -z "$PY_CV_VER" ]; then
        PY_PATH=$(which python3)
        # Cek dukungan CUDA via Python
        PY_CUDA_SUPPORT=$(python3 -c "import cv2; print(int(cv2.cuda.getCudaEnabledDeviceCount() > 0))" 2>/dev/null)
        PY_CUDA_STATUS="No"
        if [ "$PY_CUDA_SUPPORT" == "1" ]; then
            PY_CUDA_STATUS="Yes"
        fi

        check_status "OpenCV (Python detected)" "$PY_CV_VER" "System Python Libs" "WARNING" "CUDA: $PY_CUDA_STATUS|Note: 'opencv4.pc' not found in pkg-config. C++ build might fail without manual path setup."
    else
        check_status "OpenCV" "-" "-" "MISSING"
    fi
fi

# 8. Cek LibTorch
# LibTorch tidak punya pkg-config, jadi cek manual di lokasi umum
LIBTORCH_STATUS="MISSING"
LIBTORCH_VER=""
LIBTORCH_PATH=""

# Lokasi-lokasi yang mungkin untuk LibTorch (prioritas: home user, lalu system)
POSSIBLE_LIBTORCH=(
    "/home/ardyseto/libtorch"
    "/usr/local/libtorch"
    "/opt/libtorch"
    "$HOME/libtorch"
)

for LIBTORCH_DIR in "${POSSIBLE_LIBTORCH[@]}"; do
    if [ -d "$LIBTORCH_DIR" ]; then
        # Cek file build-version untuk versi
        if [ -f "$LIBTORCH_DIR/build-version" ]; then
            LIBTORCH_VER=$(cat "$LIBTORCH_DIR/build-version" | tr -d '\n')
            LIBTORCH_PATH="$LIBTORCH_DIR"
            LIBTORCH_STATUS="OK"

            # Cek file build-hash untuk commit hash
            if [ -f "$LIBTORCH_DIR/build-hash" ]; then
                LIBTORCH_HASH=$(tail -n 1 "$LIBTORCH_DIR/build-hash" | tr -d '\n' | head -c 12)
            fi

            # Cek apakah CUDA support tersedia
            if [ -f "$LIBTORCH_DIR/lib/libtorch_cuda.so" ]; then
                LIBTORCH_CUDA="Yes (CUDA enabled)"
            else
                LIBTORCH_CUDA="No (CPU only)"
            fi

            break
        fi
    fi
done

# Build extra info string for LibTorch
LIBTORCH_EXTRA=""
if [ "$LIBTORCH_STATUS" == "OK" ]; then
    if [ ! -z "$LIBTORCH_HASH" ]; then
        LIBTORCH_EXTRA="Hash: $LIBTORCH_HASH"
    fi
    LIBTORCH_EXTRA="$LIBTORCH_EXTRA|CUDA: $LIBTORCH_CUDA"
fi

check_status "LibTorch (PyTorch C++ API)" "$LIBTORCH_VER" "$LIBTORCH_PATH" "$LIBTORCH_STATUS" "$LIBTORCH_EXTRA"

# 9. Cek Eigen3
# Coba via pkg-config
EIGEN_VER=$(pkg-config --modversion eigen3 2>/dev/null)
EIGEN_PATH=$(pkg-config --variable=prefix eigen3 2>/dev/null)

if [ ! -z "$EIGEN_VER" ]; then
    check_status "Eigen3 (via pkg-config)" "$EIGEN_VER" "$EIGEN_PATH" "OK"
elif [ -d "/usr/include/eigen3" ]; then
    # Manual check di header file
    MAJOR=$(grep "#define EIGEN_WORLD_VERSION" /usr/include/eigen3/Eigen/src/Core/util/Macros.h | awk '{print $3}')
    MINOR=$(grep "#define EIGEN_MAJOR_VERSION" /usr/include/eigen3/Eigen/src/Core/util/Macros.h | awk '{print $3}')
    PATCH=$(grep "#define EIGEN_MINOR_VERSION" /usr/include/eigen3/Eigen/src/Core/util/Macros.h | awk '{print $3}')
    check_status "Eigen3 (Header Scan)" "$MAJOR.$MINOR.$PATCH" "/usr/include/eigen3" "OK"
else
    check_status "Eigen3" "-" "-" "MISSING"
fi

# 10. Cek YAML-CPP
YAML_VER=$(pkg-config --modversion yaml-cpp 2>/dev/null)
YAML_PATH=$(pkg-config --variable=prefix yaml-cpp 2>/dev/null)

if [ ! -z "$YAML_VER" ]; then
    check_status "YAML-CPP" "$YAML_VER" "$YAML_PATH" "OK"
else
    # Cek via dpkg
    YAML_DPKG=$(dpkg -l | grep libyaml-cpp-dev)
    if [ ! -z "$YAML_DPKG" ]; then
        check_status "YAML-CPP" "Detected (via dpkg)" "/usr/include/yaml-cpp" "OK"
    else
        check_status "YAML-CPP" "-" "-" "MISSING"
    fi
fi

# 11. Cek nvidia-smi (NVIDIA GPU Driver)
NVIDIA_SMI_PATH=$(which nvidia-smi)
NVIDIA_DRIVER_VER=""
NVIDIA_DRIVER_STATUS="MISSING"

if [ ! -z "$NVIDIA_SMI_PATH" ]; then
    NVIDIA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [ -z "$NVIDIA_DRIVER_VER" ]; then
        NVIDIA_DRIVER_VER="Detected (version query failed)"
    fi
    NVIDIA_DRIVER_STATUS="OK"
fi

check_status "NVIDIA Driver (nvidia-smi)" "$NVIDIA_DRIVER_VER" "$NVIDIA_SMI_PATH" "$NVIDIA_DRIVER_STATUS"

# 12. Cek cuDNN
# cuDNN tidak selalu ada pkg-config, cek via dpkg dan header
CUDNN_STATUS="MISSING"
CUDNN_VER=""
CUDNN_PATH=""

# Cek via dpkg (metode utama untuk NVIDIA repo installation)
CUDNN_DPKG=$(dpkg -l | grep libcudnn8-dev | head -n1)
if [ ! -z "$CUDNN_DPKG" ]; then
    CUDNN_VER=$(echo $CUDNN_DPKG | awk '{print $3}' | cut -d'-' -f1)
    CUDNN_PATH="/usr/include/x86_64-linux-gnu"
    CUDNN_STATUS="OK"
    METHOD="dpkg"
else
    # Fallback: cek header di lokasi umum
    POSSIBLE_CUDNN=(
        "/usr/include/cudnn_version.h"
        "/usr/include/x86_64-linux-gnu/cudnn_version.h"
        "/usr/local/cuda/include/cudnn_version.h"
        "/home/ardyseto/cudnn/include/cudnn_version.h"
    )

    for HEADER in "${POSSIBLE_CUDNN[@]}"; do
        if [ -f "$HEADER" ]; then
            MAJOR=$(grep "#define CUDNN_MAJOR" $HEADER 2>/dev/null | awk '{print $3}')
            MINOR=$(grep "#define CUDNN_MINOR" $HEADER 2>/dev/null | awk '{print $3}')
            PATCH=$(grep "#define CUDNN_PATCHLEVEL" $HEADER 2>/dev/null | awk '{print $3}')

            if [ ! -z "$MAJOR" ]; then
                CUDNN_VER="$MAJOR.$MINOR.$PATCH"
                CUDNN_PATH=$(dirname "$HEADER")
                CUDNN_STATUS="OK"
                METHOD="Header Scan"
                break
            fi
        fi
    done
fi

check_status "cuDNN (NVIDIA CUDA Deep Neural Network)" "$CUDNN_VER" "$CUDNN_PATH" "$CUDNN_STATUS"

# 13. Cek NVIDIA TensorRT
# TensorRT agak tricky, seringkali tidak ada di pkg-config standar.
# Cek dpkg
TRT_PKG=$(dpkg -l | grep libnvinfer-dev | head -n1)
TRT_STATUS="MISSING"
TRT_VER=""
TRT_PATH=""

if [ ! -z "$TRT_PKG" ]; then
    TRT_VER=$(echo $TRT_PKG | awk '{print $3}')
    TRT_PATH="/usr/include/x86_64-linux-gnu"
    TRT_STATUS="OK"
    METHOD="dpkg"
else
    # Cek Header Manual (Common locations including custom installation)
    POSSIBLE_HEADERS=(
        "/usr/include/x86_64-linux-gnu/NvInferVersion.h" 
        "/usr/include/NvInferVersion.h" 
        "/usr/local/cuda/include/NvInferVersion.h"
        "/home/ardyseto/tensorrt/include/NvInferVersion.h"  # Custom installation path
        "/opt/tensorrt/include/NvInferVersion.h"           # Common alternative path
    )

    for HEADER in "${POSSIBLE_HEADERS[@]}"; do
        if [ -f "$HEADER" ]; then
            # Extract the actual values from the defines
            MAJOR=$(grep "#define NV_TENSORRT_MAJOR" $HEADER | awk '{print $3}' | sed 's/TRT_MAJOR_ENTERPRISE//' | sed 's/[()]//g')
            MINOR=$(grep "#define NV_TENSORRT_MINOR" $HEADER | awk '{print $3}' | sed 's/TRT_MINOR_ENTERPRISE//' | sed 's/[()]//g')
            PATCH=$(grep "#define NV_TENSORRT_PATCH" $HEADER | awk '{print $3}' | sed 's/TRT_PATCH_ENTERPRISE//' | sed 's/[()]//g')
            BUILD=$(grep "#define NV_TENSORRT_BUILD" $HEADER | awk '{print $3}' | sed 's/TRT_BUILD_ENTERPRISE//' | sed 's/[()]//g')
            
            # Handle the enterprise version macros
            if [[ "$MAJOR" == "" || "$MAJOR" == *"TRT"* ]]; then
                MAJOR=$(grep "#define TRT_MAJOR_ENTERPRISE" $HEADER | awk '{print $3}')
                MINOR=$(grep "#define TRT_MINOR_ENTERPRISE" $HEADER | awk '{print $3}')
                PATCH=$(grep "#define TRT_PATCH_ENTERPRISE" $HEADER | awk '{print $3}')
                BUILD=$(grep "#define TRT_BUILD_ENTERPRISE" $HEADER | awk '{print $3}')
            fi
            
            TRT_VER="$MAJOR.$MINOR.$PATCH.$BUILD"
            TRT_PATH=$(dirname "$HEADER")
            TRT_STATUS="OK"
            TRT_NOTE=""
            METHOD="Header Scan"

            # Check if this is the custom installation
            if [[ "$HEADER" == "/home/ardyseto/tensorrt/include/NvInferVersion.h" ]]; then
                TRT_NOTE="Note: Using custom TensorRT installation at /home/ardyseto/tensorrt"
            fi
            break
        fi
    done
fi

check_status "NVIDIA TensorRT" "$TRT_VER" "$TRT_PATH" "$TRT_STATUS" "$TRT_NOTE"

# 13. Cek Python TensorRT Module
PYTHON_TRT_STATUS="MISSING"
PYTHON_TRT_VER=""
PYTHON_TRT_PATH=""

if python3 -c "import tensorrt" &> /dev/null; then
    PYTHON_TRT_VER=$(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null)
    PYTHON_TRT_PATH=$(python3 -c "import tensorrt; import os; print(os.path.dirname(tensorrt.__file__))" 2>/dev/null)
    PYTHON_TRT_STATUS="OK"

    # Check LD_LIBRARY_PATH for TensorRT
    PYTHON_TRT_NOTE=""
    if [[ ":$LD_LIBRARY_PATH:" == *":/home/ardyseto/tensorrt/lib:"* ]]; then
        PYTHON_TRT_NOTE="Note: TensorRT LD_LIBRARY_PATH correctly set"
    else
        PYTHON_TRT_NOTE="Note: TensorRT LD_LIBRARY_PATH may need to be set to /home/ardyseto/tensorrt/lib"
    fi
fi

check_status "Python TensorRT Module" "$PYTHON_TRT_VER" "$PYTHON_TRT_PATH" "$PYTHON_TRT_STATUS" "$PYTHON_TRT_NOTE"

# 14. Cek Pangolin
PANGOLIN_PATH="/home/ardyseto/Pangolin"
if [ -d "$PANGOLIN_PATH" ]; then
    PANGOLIN_VER="Detected (Unknown Version)"
    # Coba cari versi di CMakeLists.txt
    if [ -f "$PANGOLIN_PATH/CMakeLists.txt" ]; then
        MAJOR=$(grep "set(PANGOLIN_VERSION_MAJOR" "$PANGOLIN_PATH/CMakeLists.txt" | awk '{print $2}' | tr -d ')')
        MINOR=$(grep "set(PANGOLIN_VERSION_MINOR" "$PANGOLIN_PATH/CMakeLists.txt" | awk '{print $2}' | tr -d ')')
        if [ ! -z "$MAJOR" ] && [ ! -z "$MINOR" ]; then
             PANGOLIN_VER="$MAJOR.$MINOR"
        fi
    fi
    check_status "Pangolin" "$PANGOLIN_VER" "$PANGOLIN_PATH" "OK"
else
    check_status "Pangolin" "-" "-" "MISSING"
fi

# 15. Cek g2o (Thirdparty)
G2O_PATH="$(pwd)/thirdparty/g2o"
if [ -d "$G2O_PATH" ]; then
    G2O_VER="Detected (Unknown Version)"
    if [ -f "$G2O_PATH/CMakeLists.txt" ]; then
        # Use space after G2O_VERSION to avoid matching G2O_VERSION_CONFIG
        VER_STR=$(grep "set(G2O_VERSION " "$G2O_PATH/CMakeLists.txt" | awk '{print $2}' | tr -d ')')
        if [ ! -z "$VER_STR" ]; then
             G2O_VER="$VER_STR"
        fi
    fi
    check_status "g2o (Thirdparty)" "$G2O_VER" "$G2O_PATH" "OK"
else
    check_status "g2o (Thirdparty)" "-" "-" "MISSING"
fi

# 16. Cek Sophus (Thirdparty)
SOPHUS_PATH="$(pwd)/thirdparty/Sophus"
if [ -d "$SOPHUS_PATH" ]; then
    SOPHUS_VER="Detected (Unknown Version)"
    if [ -f "$SOPHUS_PATH/SOPHUS_VERSION" ]; then
        VER_STR=$(cat "$SOPHUS_PATH/SOPHUS_VERSION" | tr -d '\n')
        if [ ! -z "$VER_STR" ]; then
             SOPHUS_VER="$VER_STR"
        fi
    fi
    check_status "Sophus (Thirdparty)" "$SOPHUS_VER" "$SOPHUS_PATH" "OK"
else
    check_status "Sophus (Thirdparty)" "-" "-" "MISSING"
fi

# 17. Cek Glog (Thirdparty)
GLOG_PATH="$(pwd)/thirdparty/glog"
if [ -d "$GLOG_PATH" ]; then
    GLOG_VER="Detected (Unknown Version)"
    if [ -f "$GLOG_PATH/CMakeLists.txt" ]; then
        # Searching for "VERSION 0.8.0" (case insensitive) inside project() command usually
        # grep for line with "VERSION" and NOT "cmake_minimum_required"
        VER_STR=$(grep -i "VERSION [0-9]" "$GLOG_PATH/CMakeLists.txt" | grep -v "cmake_minimum_required" | head -n 1 | awk '{print $2}')
        
        if [ ! -z "$VER_STR" ]; then
             GLOG_VER="$VER_STR"
        fi
    fi
    check_status "Glog (Thirdparty)" "$GLOG_VER" "$GLOG_PATH" "OK"
else
    check_status "Glog (Thirdparty)" "-" "-" "MISSING"
fi

# 18. Cek CSparse (System/SuiteSparse)
CSPARSE_PATH="/usr/include/suitesparse"
if [ -d "$CSPARSE_PATH" ]; then
    # Try to find version from dpkg first as headers often lack explicit version defines
    CSPARSE_VER=""
    SUITESPARSE_PKG=$(dpkg -l | grep libsuitesparse-dev | awk '{print $3}')
    
    if [ ! -z "$SUITESPARSE_PKG" ]; then
        CSPARSE_VER="$SUITESPARSE_PKG (via dpkg)"
    else
        CSPARSE_VER="Detected"
    fi
    
    check_status "CSparse (SuiteSparse)" "$CSPARSE_VER" "$CSPARSE_PATH" "OK"
else
    check_status "CSparse (SuiteSparse)" "-" "-" "MISSING"
fi

# 19. Cek apriltag (Thirdparty)
APRILTAG_PATH="$(pwd)/thirdparty/apriltag"
if [ -d "$APRILTAG_PATH" ]; then
    APRILTAG_VER="Detected (Unknown Version)"
    if [ -f "$APRILTAG_PATH/CMakeLists.txt" ]; then
        # Try to extract version from project(apriltag VERSION 3.4.5 ...)
        VER_STR=$(grep "project(apriltag VERSION" "$APRILTAG_PATH/CMakeLists.txt" | awk '{print $3}')
        if [ ! -z "$VER_STR" ]; then
             APRILTAG_VER="$VER_STR"
        fi
    fi
    check_status "apriltag (Thirdparty)" "$APRILTAG_VER" "$APRILTAG_PATH" "OK"
else
    check_status "apriltag (Thirdparty)" "-" "-" "MISSING"
fi

# 20. Cek realsense2 (Thirdparty)
REALSENSE_PATH="$(pwd)/thirdparty/librealsense"
if [ -d "$REALSENSE_PATH" ]; then
    REALSENSE_VER="Detected (Unknown Version)"
    # Try header file first (most reliable)
    VERSION_HEADER="$REALSENSE_PATH/include/librealsense2/rs.h"
    if [ -f "$VERSION_HEADER" ]; then
        MAJOR=$(grep "#define RS2_API_MAJOR_VERSION" "$VERSION_HEADER" | awk '{print $3}')
        MINOR=$(grep "#define RS2_API_MINOR_VERSION" "$VERSION_HEADER" | awk '{print $3}')
        PATCH=$(grep "#define RS2_API_PATCH_VERSION" "$VERSION_HEADER" | awk '{print $3}')
        if [ ! -z "$MAJOR" ] && [ ! -z "$MINOR" ] && [ ! -z "$PATCH" ]; then
             REALSENSE_VER="$MAJOR.$MINOR.$PATCH"
        fi
    elif [ -f "$REALSENSE_PATH/CMakeLists.txt" ]; then
        # Fallback to CMakeLists.txt
        MAJOR=$(grep "set(REALSENSE_VERSION_MAJOR" "$REALSENSE_PATH/CMakeLists.txt" | head -n1 | cut -d ' ' -f 2 | tr -d ')')
        MINOR=$(grep "set(REALSENSE_VERSION_MINOR" "$REALSENSE_PATH/CMakeLists.txt" | head -n1 | cut -d ' ' -f 2 | tr -d ')')
        PATCH=$(grep "set(REALSENSE_VERSION_PATCH" "$REALSENSE_PATH/CMakeLists.txt" | head -n1 | cut -d ' ' -f 2 | tr -d ')')
        
        if [ ! -z "$MAJOR" ] && [ ! -z "$MINOR" ] && [ ! -z "$PATCH" ]; then
             REALSENSE_VER="$MAJOR.$MINOR.$PATCH"
        fi
    fi
    check_status "realsense2 (Thirdparty)" "$REALSENSE_VER" "$REALSENSE_PATH" "OK"
else
    check_status "realsense2 (Thirdparty)" "-" "-" "MISSING"
fi

echo -e "${BLUE}=============================================================${NC}"
echo -e "${BLUE}   Check Completed.                                          ${NC}"
echo -e "${BLUE}=============================================================${NC}"