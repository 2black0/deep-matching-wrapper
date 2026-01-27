# C++ Matchers

This directory contains C++ implementations of deep feature matchers using LibTorch (TorchScript) and OpenCV. All matchers support both CPU and CUDA execution.

## Available Matchers

| Matcher | Description | Variants | Speed | Accuracy |
|---------|-------------|----------|-------|----------|
| [XFeat](xfeat/) | Accelerated sparse/semi-dense features | `xfeat`, `xfeat-star` | Fast-Medium | Good-Better |
| [LiftFeat](liftfeat/) | Learned Invariant Feature Transform | - | Medium | Good |
| [CLIDD](clidd/) | Compact Learned Invariant Deep Descriptors | `u128`, `a48` | Fast | Good |
| [SuperPoint-LightGlue](lightglue/) | Self-supervised keypoints with transformer matching | - | Slow | Best |

> **Note:** EDM (Efficient Dense Matching) is only available in Python (`matcher/edm/`) due to performance considerations. Python implementation runs at ~35-40ms, while C++ TorchScript was 10x slower (~390ms).

## Quick Start

### Prerequisites

1. **LibTorch** (PyTorch C++ API, version 2.10.0+)
   - Download from: https://pytorch.org/get-started/locally/
   - Extract to `/path/to/libtorch`

2. **CUDA Toolkit** (optional, for GPU acceleration)
   - CUDA 13.0+ recommended
   - Install from: https://developer.nvidia.com/cuda-downloads

3. **OpenCV** (4.x or later)
   ```bash
   # Ubuntu/Debian
   sudo apt install libopencv-dev
   
   # macOS
   brew install opencv
   ```

4. **CMake** (3.16+)
   ```bash
   sudo apt install cmake
   ```

### Environment Setup

Before building, ensure pixi environment variables don't interfere with the build:

```bash
# Use a clean environment for building
env -i PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  bash
```

### General Build Process

Each matcher follows the same build pattern:

#### Step 1: Export TorchScript Models

```bash
# Example for XFeat
pixi run python matcher/xfeat/torchscript/convert_torchscript_xfeat.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096

# See individual matcher READMEs for specific export commands
```

#### Step 2: Build C++ Code

```bash
# Set LibTorch path
export LIBTORCH_DIR="/home/ardyseto/libtorch"
export CUDA_PATH="/usr/local/cuda-13.0"

# Configure CMake
cmake -S matcher-cpp/<matcher> -B matcher-cpp/<matcher>/build \
  -DLIBTORCH_DIR="$LIBTORCH_DIR" \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
  -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
  -DCaffe2_DIR="$LIBTORCH_DIR/share/cmake/Caffe2" \
  -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"

# Build
cmake --build matcher-cpp/<matcher>/build -j8
```

#### Step 3: Run Demo

```bash
# Set runtime library path
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Run demo
./matcher-cpp/<matcher>/build/demo_<matcher> \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32
```

## Device Selection

All matchers support automatic device selection:

- **CPU**: `--device cpu`
  - Uses CPU-exported TorchScript models
  - No CUDA required
  
- **CUDA**: `--device cuda`
  - Uses CUDA-exported TorchScript models (for xfeat-star)
  - Automatically selected based on device parameter
  - Requires CUDA toolkit and compatible GPU

### Important: Device-Specific Models

Some matchers require separate model exports for CPU and CUDA:

- **XFeat-Star**: Exports `xfeat_star_fp32_k4096.pt` (CPU) and `xfeat_star_fp32_k4096_cuda.pt` (CUDA)

The C++ code automatically selects the correct model based on the `--device` parameter.

## Build All Matchers

```bash
#!/bin/bash
# Build all matchers at once

export LIBTORCH_DIR="/home/ardyseto/libtorch"
export CUDA_PATH="/usr/local/cuda-13.0"

for matcher in xfeat liftfeat clidd lightglue; do
  echo "Building $matcher..."
  
  cmake -S matcher-cpp/$matcher -B matcher-cpp/$matcher/build \
    -DLIBTORCH_DIR="$LIBTORCH_DIR" \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
    -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
    -DCaffe2_DIR="$LIBTORCH_DIR/share/cmake/Caffe2" \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"
  
  cmake --build matcher-cpp/$matcher/build -j8
done

echo "All matchers built successfully!"
```

## Troubleshooting

### LibTBB Conflicts

If you see errors like `undefined reference to __cxa_call_terminate`, this is caused by pixi's libtbb conflicting with system libraries.

**Solution**: Build in a clean environment without pixi variables:

```bash
env -i PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake -S matcher-cpp/<matcher> -B matcher-cpp/<matcher>/build [...]
```

### Missing TorchScript Weights

```
Missing TorchScript weights: matcher-cpp/<matcher>/weights/<model>.pt
```

**Solution**: Export the model first (see Step 1 above and individual matcher READMEs).

### CUDA Device Mismatch

If you exported models on CPU but try to run on CUDA (or vice versa):

**Solution**: 
- For xfeat-star: Export both CPU and CUDA versions
- For other matchers: Re-export on the target device or use CPU

### CMake Cannot Find LibTorch

```
Could not find Torch
```

**Solution**: Set `LIBTORCH_DIR` and all related paths:

```bash
cmake -DLIBTORCH_DIR=/path/to/libtorch \
      -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
      -DCaffe2_DIR=/path/to/libtorch/share/cmake/Caffe2 \
      [...]
```

## Performance Benchmarks

Test images: `assets/ref.png` (800x600) and `assets/tgt.png` (800x600)

| Matcher | Device | Matches | Inliers | Inference Time (ms) |
|---------|--------|---------|---------|---------------------|
| XFeat | CPU | 54 | 9 | ~50 |
| XFeat | CUDA | 54 | 9 | ~15 |
| XFeat-Star | CPU | 653 | 347 | ~215 |
| XFeat-Star | CUDA | 650 | 349 | ~192 |
| LiftFeat | CPU | 735 | 253 | ~300 |
| LiftFeat | CUDA | 735 | 255 | ~40 |
| CLIDD-U128 | CPU | 865 | 669 | ~280 |
| CLIDD-U128 | CUDA | 864 | 666 | ~25 |
| CLIDD-A48 | CPU | 865 | 669 | ~260 |
| CLIDD-A48 | CUDA | 864 | 666 | ~22 |
| SuperPoint-LightGlue | CPU | 455 | 447 | ~2304 |
| SuperPoint-LightGlue | CUDA | 455 | 450 | ~1747 |

*Note: Times measured on NVIDIA RTX 4090 / AMD Ryzen 9 7950X*

**EDM Performance (Python only):**
- Python eager mode: ~35ms CUDA
- Python ONNX Runtime: ~40ms CUDA
- C++ TorchScript was 10x slower, hence not included in C++

## API Usage

All matchers provide a consistent C++ API:

```cpp
#include "<matcher>/<Matcher>TorchMatcher.h"

// Configure matcher
dmw::<matcher>::<Matcher>Config cfg;
cfg.device = "cuda";
cfg.dtype = "fp32";
cfg.top_k = 4096;  // or appropriate value for matcher
// ... matcher-specific options

// Create matcher
dmw::<matcher>::<Matcher>TorchMatcher matcher(cfg);

// Match two images
cv::Mat img0 = cv::imread("image1.jpg");
cv::Mat img1 = cv::imread("image2.jpg");
auto result = matcher.match(img0, img1);

// Access results
std::cout << "Matches: " << result.matched_kpts0.size() << "\n";
std::cout << "Inliers: " << result.inlier_kpts0.size() << "\n";
std::cout << "Inference time: " << result.ms_infer << " ms\n";
```

## Individual Matcher Documentation

For detailed information about each matcher:

- [XFeat](xfeat/README.md) - Sparse and semi-dense features
- [LiftFeat](liftfeat/README.md) - Learned invariant features
- [CLIDD](clidd/README.md) - Compact learned descriptors
- [SuperPoint-LightGlue](lightglue/README.md) - Self-supervised keypoints with transformer matching

> **Note:** For EDM (dense matching), use the Python implementation in `matcher/edm/`

## Contributing

When adding new matchers:

1. Follow the directory structure: `matcher-cpp/<matcher>/`
2. Use consistent naming: `<Matcher>TorchMatcher.{h,cpp}`
3. Provide TorchScript export scripts in `matcher/<matcher>/torchscript/`
4. Include comprehensive README.md with build and usage instructions
5. Add demo executable: `demo_<matcher>.cpp`

## License

Each matcher implementation follows the license of its original project. See individual matcher directories for details.
