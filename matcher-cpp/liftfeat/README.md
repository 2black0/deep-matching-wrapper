# LiftFeat C++ Implementation

This directory contains the C++ implementation of LiftFeat (Learned Invariant Feature Transform) using LibTorch (TorchScript) and OpenCV.

LiftFeat combines learned feature detection with classical LIFT-inspired descriptor extraction for robust image matching.

## Directory Structure

```
matcher-cpp/liftfeat/
├── CMakeLists.txt                    # Build configuration
├── include/liftfeat/
│   └── LiftFeatTorchMatcher.h       # C++ header
├── src/
│   └── LiftFeatTorchMatcher.cpp     # C++ implementation
├── demo/
│   └── demo_liftfeat.cpp            # Demo executable
└── weights/                         # TorchScript models (generated)
    └── liftfeat_fp32_k4096.pt       # LiftFeat model (CPU/CUDA)
```

## Prerequisites

1. **LibTorch** (PyTorch C++ API, version 2.10.0+)
2. **OpenCV** (4.x or later)
3. **CMake** (3.16+)
4. **C++17 compatible compiler**
5. **CUDA Toolkit** (13.0+, optional for GPU acceleration)

## What Runs Where

- **TorchScript (LibTorch, CUDA/CPU)**
  - LiftFeat forward pass (feature extraction + NMS + top-k selection + descriptor sampling)
  - Mutual nearest-neighbor descriptor matching (cosine similarity)
- **OpenCV (CPU)**
  - `cv::findHomography(..., cv::USAC_MAGSAC, ...)` for homography estimation and inlier mask
  - Visualization output (`result.jpg`)

## Step 1: Export TorchScript Weights

LibTorch cannot directly load Python `.pth` state_dict files, so export a TorchScript `.pt` model:

```bash
pixi run python matcher/liftfeat/torchscript/convert_torchscript.py \
  --topk 4096 \
  --detect-threshold 0.05
```

Output: `matcher-cpp/liftfeat/weights/liftfeat_fp32_k4096.pt`

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
  cmake -S matcher-cpp/liftfeat -B matcher-cpp/liftfeat/build \
    -DLIBTORCH_DIR="$LIBTORCH_DIR" \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
    -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
    -DCaffe2_DIR="$LIBTORCH_DIR/share/cmake/Caffe2" \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"

# Build
env -i PATH="$CUDA_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake --build matcher-cpp/liftfeat/build -j8
```

This creates:
- `libdmw_liftfeat_lib.a` - Static library
- `demo_liftfeat` - Demo executable

## Step 3: Run the Demo

### CPU Execution

```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/liftfeat/build/demo_liftfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cpu \
  --dtype fp32 \
  --topk 4096 \
  --output yes
```

### CUDA Execution

```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/liftfeat/build/demo_liftfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --topk 4096 \
  --output yes
```

Output files (when `--output yes`):
- `outputs/matching-cpp/liftfeat_fp32_ref_tgt/result.jpg` - Visualization
- `outputs/matching-cpp/liftfeat_fp32_ref_tgt/result.txt` - Match statistics

## Configuration Options

- `--device`: Device for inference (`cpu` or `cuda`)
- `--dtype`: Data type (currently only `fp32` supported)
- `--topk`: Number of top keypoints to keep (must match exported model, default: 4096)
- `--detect-threshold`: Threshold for keypoint detection (default: 0.05)
- `--output`: Save output visualization (`yes` or `no`)
- `--out`: Custom output path for result image

## Performance Benchmarks

Test images: `assets/ref.png` (800x600) and `assets/tgt.png` (800x600)

| Device | Total Keypoints (per image) | Matches | Inliers | Inlier Ratio | Inference Time (ms) |
|--------|------------------------------|---------|---------|--------------|---------------------|
| CPU | 4096 | 735 | 253 | 34.4% | ~300 |
| CUDA | 4096 | 735 | 255 | 34.7% | ~40 |

*Measured on NVIDIA RTX 4090 / AMD Ryzen 9 7950X*

**CUDA speedup**: ~7x faster than CPU

## Python vs C++ Parity

With the default settings (`topk=4096`, `detect-threshold=0.05`), the C++ outputs should be very close to Python:

**Python:**
```bash
pixi run python demo_matcher.py --matcher liftfeat \
  --img1 assets/ref.png --img2 assets/tgt.png
```

**C++:**
```bash
./matcher-cpp/liftfeat/build/demo_liftfeat \
  --img1 assets/ref.png --img2 assets/tgt.png \
  --device cuda --dtype fp32 --topk 4096
```

Expected parity:
- Total keypoints: Exact match (4096 per image)
- Matched keypoints: Typically ±0-2 difference
- Inliers: Typically ±0-2 difference

## API Usage

```cpp
#include "liftfeat/LiftFeatTorchMatcher.h"

// Configure LiftFeat matcher
dmw::liftfeat::LiftFeatConfig cfg;
cfg.device = "cuda";
cfg.dtype = "fp32";
cfg.top_k = 4096;
cfg.detect_threshold = 0.05f;
cfg.weights_base_dir = "matcher-cpp/liftfeat/weights";

// Create matcher
dmw::liftfeat::LiftFeatTorchMatcher matcher(cfg);

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

## Troubleshooting

### Missing TorchScript weights error

```
Missing TorchScript weights: matcher-cpp/liftfeat/weights/liftfeat_fp32_k4096.pt
```

**Solution**: Run the conversion script (see Step 1).

### Different top_k value

If you exported with a different `--topk` value, use the same value when running:

```bash
./demo_liftfeat --topk 2048 ...  # Must match exported model
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
2. **Descriptor Matching**: Mutual nearest neighbors with cosine similarity
3. **Device Portability**: Same TorchScript model works on both CPU and CUDA
4. **Deterministic Output**: Results should be consistent across runs (±small numerical differences)

## License

This implementation follows the same license as the original LiftFeat project.
