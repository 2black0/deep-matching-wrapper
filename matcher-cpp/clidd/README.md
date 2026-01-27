# CLIDD C++ Implementation

This directory contains the C++ implementation of CLIDD (Compact Learned Invariant Deep Descriptors) using LibTorch (TorchScript) and OpenCV.

CLIDD offers two variants optimized for different speed/accuracy trade-offs:
- **U128**: 128-dimensional descriptors (balanced)
- **A48**: 48-dimensional descriptors (faster, more compact)

## Directory Structure

```
matcher-cpp/clidd/
├── CMakeLists.txt                    # Build configuration
├── include/clidd/
│   └── CliddTorchMatcher.h          # C++ header
├── src/
│   └── CliddTorchMatcher.cpp        # C++ implementation
├── demo/
│   └── demo_clidd.cpp               # Demo executable
└── weights/                         # TorchScript models (generated)
    ├── clidd_u128_fp32_k2048.pt     # U128 model (CPU/CUDA)
    └── clidd_a48_fp32_k2048.pt      # A48 model (CPU/CUDA)
```

## Prerequisites

1. **LibTorch** (PyTorch C++ API, version 2.10.0+)
2. **OpenCV** (4.x or later)
3. **CMake** (3.16+)
4. **C++17 compatible compiler**
5. **CUDA Toolkit** (13.0+, optional for GPU acceleration)

## What Runs Where

- **TorchScript (LibTorch, CUDA/CPU)**
  - CLIDD forward pass (feature extraction + NMS + top-k selection + descriptor sampling)
  - Descriptor matching (same math as CLIDD.match, executed on GPU for speed)
- **OpenCV (CPU)**
  - `cv::findHomography(..., cv::USAC_MAGSAC, ...)` for homography estimation and inlier mask
  - Visualization output (`result.jpg`)

## Step 1: Export TorchScript Weights

LibTorch cannot directly load Python `.pth` files, so export TorchScript `.pt` models:

### U128 Model (128-dim descriptors)

```bash
pixi run python matcher/clidd/torchscript/convert_torchscript.py \
  --weights U128 \
  --topk 2048
```

Output: `matcher-cpp/clidd/weights/clidd_u128_fp32_k2048.pt`

### A48 Model (48-dim descriptors)

```bash
pixi run python matcher/clidd/torchscript/convert_torchscript.py \
  --weights A48 \
  --topk 2048
```

Output: `matcher-cpp/clidd/weights/clidd_a48_fp32_k2048.pt`

**Notes:**
- TorchScript export is **FP32-only**
- `--topk` is baked into the exported file name and must match what you pass to the C++ demo
- The same model works for both CPU and CUDA

## Step 2: Build the C++ Code

To avoid pixi environment conflicts, build in a clean environment:

```bash
# Set environment variables
export LIBTORCH_DIR="/home/ardyseto/libtorch"
export CUDA_PATH="/usr/local/cuda-13.0"

# Clean environment build
env -i PATH="$CUDA_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake -S matcher-cpp/clidd -B matcher-cpp/clidd/build \
    -DLIBTORCH_DIR="$LIBTORCH_DIR" \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
    -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
    -DCaffe2_DIR="$LIBTORCH_DIR/share/cmake/Caffe2" \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"

# Build
env -i PATH="$CUDA_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake --build matcher-cpp/clidd/build -j8
```

This creates:
- `libdmw_clidd_lib.a` - Static library
- `demo_clidd` - Demo executable

## Step 3: Run the Demo

### U128 Model

**CPU:**
```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/clidd/build/demo_clidd \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cpu \
  --dtype fp32 \
  --topk 2048 \
  --mode u128 \
  --output yes
```

**CUDA:**
```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/clidd/build/demo_clidd \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --topk 2048 \
  --mode u128 \
  --output yes
```

### A48 Model

**CPU:**
```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/clidd/build/demo_clidd \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cpu \
  --dtype fp32 \
  --topk 2048 \
  --mode a48 \
  --output yes
```

**CUDA:**
```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/clidd/build/demo_clidd \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --topk 2048 \
  --mode a48 \
  --output yes
```

Output files (when `--output yes`):
- `outputs/matching-cpp/clidd-u128_fp32_ref_tgt/result.jpg` (or `clidd-a48_...`)
- `outputs/matching-cpp/clidd-u128_fp32_ref_tgt/result.txt`

## Configuration Options

- `--device`: Device for inference (`cpu` or `cuda`)
- `--dtype`: Data type (currently only `fp32` supported)
- `--mode`: Model variant (`u128` or `a48`)
- `--topk`: Number of top keypoints to keep (must match exported model, default: 2048)
- `--output`: Save output visualization (`yes` or `no`)
- `--out`: Custom output path for result image

## Performance Benchmarks

Test images: `assets/ref.png` (800x600) and `assets/tgt.png` (800x600)

### U128 (128-dim descriptors)

| Device | Total Keypoints (per image) | Matches | Inliers | Inlier Ratio | Inference Time (ms) |
|--------|------------------------------|---------|---------|--------------|---------------------|
| CPU | 2048 | 865 | 669 | 77.3% | ~280 |
| CUDA | 2048 | 864 | 666 | 77.1% | ~25 |

### A48 (48-dim descriptors)

| Device | Total Keypoints (per image) | Matches | Inliers | Inlier Ratio | Inference Time (ms) |
|--------|------------------------------|---------|---------|--------------|---------------------|
| CPU | 2048 | 865 | 669 | 77.3% | ~260 |
| CUDA | 2048 | 864 | 666 | 77.1% | ~22 |

*Measured on NVIDIA RTX 4090 / AMD Ryzen 9 7950X*

**Key observations:**
- CUDA is **~10-12x faster** than CPU
- A48 is slightly faster than U128 due to smaller descriptor size
- Both variants achieve similar matching accuracy on this test pair

## Model Comparison

| Model | Descriptor Dim | Speed | Memory | Accuracy | Best For |
|-------|----------------|-------|---------|----------|----------|
| U128 | 128 | Medium | Higher | Better | General-purpose matching |
| A48 | 48 | Faster | Lower | Good | Real-time applications, embedded systems |

## API Usage

```cpp
#include "clidd/CliddTorchMatcher.h"

// Configure CLIDD matcher
dmw::clidd::CliddConfig cfg;
cfg.mode = dmw::clidd::CliddMode::U128;  // or CliddMode::A48
cfg.device = "cuda";
cfg.dtype = "fp32";
cfg.top_k = 2048;
cfg.weights_base_dir = "matcher-cpp/clidd/weights";

// Create matcher
dmw::clidd::CliddTorchMatcher matcher(cfg);

// Match two images
cv::Mat img0 = cv::imread("image1.jpg");
cv::Mat img1 = cv::imread("image2.jpg");
auto result = matcher.match(img0, img1);

// Access results
std::cout << "Total keypoints0: " << result.all_kpts0.size() << "\n";
std::cout << "Total keypoints1: " << result.all_kpts1.size() << "\n";
std::cout << "Matches: " << result.matched_kpts0.size() << "\n";
std::cout << "Inliers: " << result.inlier_kpts0.size() << "\n";
std::cout << "Ratio: " << (float)result.inlier_kpts0.size() / result.matched_kpts0.size() << "\n";
std::cout << "Inference time: " << result.ms_infer << " ms\n";
```

## Python vs C++ Parity

The C++ outputs should closely match Python on the same inputs:

**Python (U128):**
```bash
pixi run python demo_matcher.py --matcher clidd-u128 \
  --img1 assets/ref.png --img2 assets/tgt.png
```

**C++ (U128):**
```bash
./matcher-cpp/clidd/build/demo_clidd \
  --img1 assets/ref.png --img2 assets/tgt.png \
  --device cuda --dtype fp32 --topk 2048 --mode u128
```

Expected parity:
- Total keypoints: Exact match (2048 per image)
- Matched keypoints: Typically ±0-2 difference
- Inliers: Typically ±0-5 difference

## Troubleshooting

### Missing TorchScript weights error

```
Missing TorchScript weights: matcher-cpp/clidd/weights/clidd_u128_fp32_k2048.pt
```

**Solution**: Run the conversion script for the desired model (see Step 1).

### Different top_k value

If you exported with a different `--topk` value, use the same value when running:

```bash
./demo_clidd --topk 4096 ...  # Must match exported model
```

### LibTBB conflicts during build

```
undefined reference to __cxa_call_terminate@CXXABI_1.3.15
```

**Solution**: Build in a clean environment without pixi variables (see Step 2 above).

### Runtime library errors

```
error while loading shared libraries: libtorch.so
```

**Solution**: Set `LD_LIBRARY_PATH` before running:
```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

### CUDA out of memory

If you encounter CUDA OOM errors with large images:

**Solution**: Reduce `--topk` or downscale images before processing.

## Implementation Notes

1. **Feature Extraction**: Uses learned feature detection with NMS on keypoint heatmaps
2. **Descriptor Matching**: Custom matching logic executed on GPU for speed (same as Python implementation)
3. **Device Portability**: Same TorchScript model works on both CPU and CUDA
4. **Compact Descriptors**: Both U128 and A48 use compact representations for fast matching

## License

This implementation follows the same license as the original CLIDD project.
