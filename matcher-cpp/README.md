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

This project uses [pixi](https://pixi.sh/) for dependency management, which handles most dependencies automatically.

1. **System CUDA Toolkit** (for GPU acceleration)
   - CUDA 13.0.88 (system-installed)
   - Location: `/usr/local/cuda-13.0/`

2. **LibTorch** (PyTorch C++ API, version 2.10.0+cu130)
   - Download from: https://pytorch.org/get-started/locally/
   - Extract to `/home/ardyseto/libtorch/`
   - Pre-installed for this project

3. **System OpenCV** (4.13.0 with CUDA support)
   - Pre-installed at `/usr/local/lib/`
   - Symlinked to pixi environment for Python

4. **Pixi environment** (includes CMake, compilers, etc.)
   ```bash
   # Install pixi if needed
   curl -fsSL https://pixi.sh/install.sh | bash
   
   # Install dependencies
   pixi install
   ```

### Build Using Pixi (Recommended)

The easiest way to build all matchers:

```bash
# Build all matchers
pixi run build-all-cpp

# Or build individually
pixi run build-xfeat
pixi run build-liftfeat
pixi run build-clidd

# Clean builds
pixi run clean-cpp
```

These pixi tasks are defined in `pixi.toml` and internally call:
- `scripts/build_all_matchers_cpp.sh` - Builds all matchers sequentially
- `scripts/build_matcher_cpp.sh <matcher>` - Builds a single matcher

The scripts automatically:
- Use system CUDA 13.0.88 at `/usr/local/cuda-13.0/`
- Link LibTorch 2.10.0+cu130 from `/home/ardyseto/libtorch/`
- Link system OpenCV 4.13.0 with CUDA support
- Set correct library paths for OpenCV linking
- Configure CMake with all necessary flags

**Build Scripts:**
- **`scripts/build_matcher_cpp.sh`** - Single matcher builder
  ```bash
  bash scripts/build_matcher_cpp.sh xfeat    # Build only XFeat
  bash scripts/build_matcher_cpp.sh liftfeat # Build only LiftFeat
  bash scripts/build_matcher_cpp.sh clidd    # Build only CLIDD
  ```
  
- **`scripts/build_all_matchers_cpp.sh`** - All matchers builder
  ```bash
  bash scripts/build_all_matchers_cpp.sh     # Build all matchers
  ```
  This script loops through all matchers and calls `build_matcher_cpp.sh` for each one.

### Manual Build Process

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
# Using pixi (recommended)
pixi run build-<matcher>  # e.g., pixi run build-xfeat

# Or manually:
export LIBTORCH_DIR="/home/ardyseto/libtorch"
export CUDA_PATH="/usr/local/cuda-13.0"
export OPENCV_LIB_PATH="/usr/local/lib"

# Configure CMake
cmake -S matcher-cpp/<matcher> -B matcher-cpp/<matcher>/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBTORCH_DIR="$LIBTORCH_DIR" \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
  -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
  -DCaffe2_DIR="$LIBTORCH_DIR/share/cmake/Caffe2" \
  -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$OPENCV_LIB_PATH" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$OPENCV_LIB_PATH"

# Build
cmake --build matcher-cpp/<matcher>/build -j$(nproc)
```

#### Step 3: Run Demo

```bash
# Set runtime library path
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:/usr/local/cuda-13.0/lib64:/usr/local/lib:$LD_LIBRARY_PATH

# Run demo
./matcher-cpp/<matcher>/build/demo_<matcher> \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32
```

### Current Build Status

**System Configuration:**
- CUDA: 13.0.88 (system at `/usr/local/cuda-13.0/`)
- LibTorch: 2.10.0+cu130 (at `/home/ardyseto/libtorch/`)
- OpenCV: 4.13.0 with CUDA support (at `/usr/local/lib/`)
- Python CUDA: 12.9 via pixi (independent, for Python only)

**All matchers tested on:** NVIDIA RTX 4060 Ti 8GB (SM 8.9)

| Matcher | Build Status | CUDA 13.0 | Test Status | Demo Time |
|---------|--------------|-----------|-------------|-----------|
| XFeat | ✅ Built | ✅ Yes | ✅ Passed | 259.6ms |
| LiftFeat | ✅ Built | ✅ Yes | ✅ Passed | 41.0ms |
| CLIDD | ✅ Built | ✅ Yes | ✅ Passed | 15.1ms |

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

Using pixi (recommended):
```bash
pixi run build-all-cpp
```

Or directly with the build script:
```bash
bash scripts/build_all_matchers_cpp.sh
```

This will build all matchers (xfeat, liftfeat, clidd) sequentially using the configuration from `scripts/build_matcher_cpp.sh`.

## Troubleshooting

### OpenCV Linking Error

If you see errors like `cannot find -lopencv_core`:

**Solution**: Add OpenCV library path to linker flags:
```bash
-DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/lib" \
-DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/lib"
```

This is automatically handled by `pixi run build-*` commands.

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

Test images: `assets/ref.png` and `assets/tgt.png` (800x600)  
Hardware: NVIDIA RTX 4060 Ti 8GB, System CUDA 13.0.88, LibTorch 2.10.0+cu130

| Matcher | Keypoints | Matches | Inliers | Inlier Ratio | Inference (ms) | Match (ms) | Total (ms) |
|---------|-----------|---------|---------|--------------|----------------|------------|------------|
| **CLIDD** | 2048 | 864 | 666 | 77.1% | 6.5 | 1.1 | **15.1** |
| **LiftFeat** | 4096 | 735 | 255 | 34.7% | 19.5 | 1.0 | **41.0** |
| **XFeat** | 4096 | 832 | 154 | 18.5% | 10.9 | 236.3 | **259.6** |

**Winner: CLIDD** - Fastest (15.1ms) with best accuracy (77.1% inlier ratio)!

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
