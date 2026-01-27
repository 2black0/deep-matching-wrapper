# LiftFeat C++ Implementation

C++ implementation of LiftFeat (Learned Invariant Feature Transform) using LibTorch and OpenCV.

LiftFeat combines learned feature detection with classical descriptor extraction for robust image matching.

## Prerequisites

1. **LibTorch** (PyTorch C++ API, version 2.10.0+)
2. **OpenCV** (4.x or later)
3. **CMake** (3.16+)
4. **C++17 compatible compiler**
5. **CUDA Toolkit** (13.0+, optional for GPU acceleration)

## Workflow

### 1. Export PyTorch Model to TorchScript

```bash
pixi run python matcher/liftfeat/torchscript/convert_torchscript.py \
  --topk 4096 \
  --detect-threshold 0.05
```

This creates `matcher-cpp/liftfeat/weights/liftfeat_fp32_k4096.pt`.

### 2. Build

```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.0

cd matcher-cpp/liftfeat
rm -rf build && mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH=/home/ardyseto/libtorch \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
      ..

make -j$(nproc)
```

### 3. Run Demo

**CPU:**
```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH

./matcher-cpp/liftfeat/build/demo_liftfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cpu \
  --output yes
```

**CUDA:**
```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH

./matcher-cpp/liftfeat/build/demo_liftfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --output yes
```

## Options

- `--device`: Device (`cpu` or `cuda`)
- `--dtype`: Data type (`fp32` only)
- `--topk`: Number of top keypoints (default: 4096, must match exported model)
- `--detect-threshold`: Keypoint detection threshold (default: 0.05)
- `--output`: Save visualization (`yes` or `no`)
- `--out`: Custom output path

## Performance Benchmarks

Test images: `assets/ref.png` and `assets/tgt.png` (800x600)

| Device | Matches | Inliers | Ratio | Time |
|--------|---------|---------|-------|------|
| CPU | 735 | 253 | 34.4% | 352ms |
| CUDA | 735 | 255 | 34.7% | 41ms |

*Measured on NVIDIA RTX 4060 Ti / AMD Ryzen*

**CUDA speedup**: 8.6x faster than CPU

## Comparison with Python

C++ produces nearly identical results to Python:

| Implementation | Device | Matches | Inliers | Ratio | Time |
|----------------|--------|---------|---------|-------|------|
| C++ | CUDA | 735 | 255 | 34.7% | 41ms |
| Python | CUDA | 736 | 256 | 34.8% | 35ms |

**Status**: ✅ C++ CUDA is **17% faster** than Python CUDA!

## API Usage

```cpp
#include "liftfeat/LiftFeatTorchMatcher.h"

// Configure matcher
dmw::liftfeat::LiftFeatConfig cfg;
cfg.device = "cuda";
cfg.dtype = "fp32";
cfg.top_k = 4096;
cfg.detect_threshold = 0.05f;

// Create matcher
dmw::liftfeat::LiftFeatTorchMatcher matcher(cfg);

// Match images
cv::Mat img0 = cv::imread("image1.jpg");
cv::Mat img1 = cv::imread("image2.jpg");
auto result = matcher.match(img0, img1);

// Access results
std::cout << "Matches: " << result.matched_kpts0.size() << "\n";
std::cout << "Inliers: " << result.inlier_kpts0.size() << "\n";
```

## Troubleshooting

**Missing TorchScript weights:**
```
Missing TorchScript weights: matcher-cpp/liftfeat/weights/liftfeat_fp32_k4096.pt
```
→ Run the conversion script (see step 1).

**TBB ABI errors during build:**
```
undefined reference to __cxa_call_terminate@CXXABI_1.3.15
```
→ Already fixed in CMakeLists.txt with linker flag.

**Runtime library errors:**
```
error while loading shared libraries: libtorch.so
```
→ Set `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH
```
