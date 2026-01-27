# XFeat C++ Implementation

This directory contains the C++ implementation of XFeat feature matching with two variants:
- **XFeat**: Sparse features + Mutual Nearest Neighbors matching
- **XFeat-Star**: Semi-dense features + refinement

Based on the paper: "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024"
- Website: https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

## Directory Structure

```
matcher-cpp/xfeat/
├── CMakeLists.txt                    # Build configuration
├── include/xfeat/
│   └── XFeatTorchMatcher.h          # C++ header
├── src/
│   └── XFeatTorchMatcher.cpp        # C++ implementation
├── demo/
│   └── demo_xfeat.cpp               # Demo executable
└── weights/                         # TorchScript models (generated)
    ├── xfeat_fp32_k4096.pt          # XFeat sparse model (CPU)
    ├── xfeat_star_fp32_k4096.pt     # XFeat-Star model (CPU)
    └── xfeat_star_fp32_k4096_cuda.pt # XFeat-Star model (CUDA)
```

## Prerequisites

1. **LibTorch** (PyTorch C++ API, version 2.10.0+)
2. **OpenCV** (4.x or later)
3. **CMake** (3.16+)
4. **C++17 compatible compiler**
5. **CUDA Toolkit** (13.0+, optional for GPU acceleration)

## Step 1: Export Python Models to TorchScript

Before building the C++ code, export the PyTorch models to TorchScript format.

### 1.1 XFeat (Sparse features)

```bash
pixi run python matcher/xfeat/torchscript/convert_torchscript_xfeat.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096 \
  --detection-threshold 0.05
```

Output: `matcher-cpp/xfeat/weights/xfeat_fp32_k4096.pt`

### 1.2 XFeat-Star (Semi-dense) - CPU and CUDA versions

XFeat-Star requires separate exports for CPU and CUDA due to device-specific optimizations:

**CPU version:**
```bash
pixi run python matcher/xfeat/torchscript/convert_torchscript_xfeat_star.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096 \
  --fine-conf 0.25 \
  --device cpu
```

Output: `matcher-cpp/xfeat/weights/xfeat_star_fp32_k4096.pt`

**CUDA version:**
```bash
pixi run python matcher/xfeat/torchscript/convert_torchscript_xfeat_star.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096 \
  --fine-conf 0.25 \
  --device cuda
```

Output: `matcher-cpp/xfeat/weights/xfeat_star_fp32_k4096_cuda.pt`

**Important**: The C++ code automatically selects the correct model based on the `--device` parameter.

## Step 2: Build the C++ Code

To avoid pixi environment conflicts, build in a clean environment:

```bash
# Set environment variables
export LIBTORCH_DIR="/home/ardyseto/libtorch"
export CUDA_PATH="/usr/local/cuda-13.0"

# Clean environment build
env -i PATH="$CUDA_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake -S matcher-cpp/xfeat -B matcher-cpp/xfeat/build \
    -DLIBTORCH_DIR="$LIBTORCH_DIR" \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
    -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
    -DCaffe2_DIR="$LIBTORCH_DIR/share/cmake/Caffe2" \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"

# Build
env -i PATH="$CUDA_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake --build matcher-cpp/xfeat/build -j8
```

This creates:
- `libdmw_xfeat_lib.a` - Static library
- `demo_xfeat` - Demo executable

## Step 3: Run the Demo

### XFeat (Sparse + MNN)

**CPU:**
```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/xfeat/build/demo_xfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --mode xfeat \
  --device cpu \
  --dtype fp32 \
  --topk 4096 \
  --min-cossim -1 \
  --output yes
```

**CUDA:**
```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/xfeat/build/demo_xfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --mode xfeat \
  --device cuda \
  --dtype fp32 \
  --topk 4096 \
  --min-cossim -1 \
  --output yes
```

### XFeat-Star (Semi-dense + Refinement)

**CPU:**
```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/xfeat/build/demo_xfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --mode xfeat-star \
  --device cpu \
  --dtype fp32 \
  --topk 4096 \
  --fine-conf 0.25 \
  --output yes
```

**CUDA:**
```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/xfeat/build/demo_xfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --mode xfeat-star \
  --device cuda \
  --dtype fp32 \
  --topk 4096 \
  --fine-conf 0.25 \
  --output yes
```

## Configuration Options

### Common Options

- `--mode`: Matching mode (`xfeat`, `xfeat-star`)
- `--device`: Device for inference (`cpu` or `cuda`)
- `--dtype`: Data type (currently only `fp32` supported)
- `--topk`: Number of top keypoints to keep (must match exported model)
- `--output`: Save output visualization (`yes` or `no`)
- `--out`: Custom output path for result image
- `--draw-all`: Draw all matches including outliers

### XFeat-specific Options

- `--detection-threshold`: Threshold for keypoint detection (default: 0.05)
- `--min-cossim`: Minimum cosine similarity for MNN matching (default: -1, disabled)

### XFeat-Star-specific Options

- `--fine-conf`: Confidence threshold for refinement (default: 0.25)

## Performance Benchmarks

Test images: `assets/ref.png` (800x600) and `assets/tgt.png` (800x600)

| Mode | Device | Matches | Inliers | Inlier Ratio | Inference Time (ms) |
|------|--------|---------|---------|--------------|---------------------|
| XFeat | CPU | 54 | 9 | 16.7% | ~50 |
| XFeat | CUDA | 54 | 9 | 16.7% | ~15 |
| XFeat-Star | CPU | 653 | 347 | 53.1% | ~215 |
| XFeat-Star | CUDA | 650 | 349 | 53.7% | ~192 |

*Measured on NVIDIA RTX 4090 / AMD Ryzen 9 7950X*

## Performance Comparison

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| XFeat | Fast | Good | Real-time applications, sparse matching |
| XFeat-Star | Medium | Better | Semi-dense correspondence, higher accuracy needs |

## Implementation Notes

1. **XFeat Mode**: Uses sparse feature extraction (NMS on keypoint heatmap) with mutual nearest neighbor matching using cosine similarity
2. **XFeat-Star Mode**: Extracts features at dual scales (0.6x and 1.3x) and refines matches using an MLP. This produces semi-dense matches.
3. **Device Selection**: XFeat-Star automatically selects the appropriate model file based on the device parameter (CPU or CUDA)

## API Usage

```cpp
#include "xfeat/XFeatTorchMatcher.h"

// Configure XFeat matcher
dmw::xfeat::XFeatConfig cfg;
cfg.mode = dmw::xfeat::XFeatMode::XFEAT;
cfg.device = "cpu";
cfg.top_k = 4096;
cfg.min_cossim = 0.82f;
cfg.weights_base_dir = "matcher-cpp/xfeat/weights";

// Create matcher
dmw::xfeat::XFeatTorchMatcher matcher(cfg);

// Match two images
cv::Mat img0 = cv::imread("image1.jpg");
cv::Mat img1 = cv::imread("image2.jpg");
auto result = matcher.match(img0, img1);

// Access results
std::cout << "Matches: " << result.matched_kpts0.size() << "\n";
std::cout << "Inliers: " << result.inlier_kpts0.size() << "\n";
std::cout << "Ratio: " << (float)result.inlier_kpts0.size() / result.matched_kpts0.size() << "\n";
std::cout << "Inference time: " << result.ms_infer << " ms\n";
```

## Troubleshooting

### Missing TorchScript weights error

```
Missing TorchScript weights: matcher-cpp/xfeat/weights/xfeat_fp32_k4096.pt
```

**Solution**: Run the conversion script for the appropriate mode (see Step 1).

### Device mismatch for XFeat-Star

If you get device-related errors with XFeat-Star:

**Solution**: Make sure you exported both CPU and CUDA versions:
- CPU: `--device cpu` → creates `xfeat_star_fp32_k4096.pt`
- CUDA: `--device cuda` → creates `xfeat_star_fp32_k4096_cuda.pt`

### Different top_k value

If you exported with a different `--topk` value, use the same value when running:

```bash
./demo_xfeat --topk 2048 ...  # Must match exported model
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

## Citation

If you use XFeat in your research, please cite:

```bibtex
@inproceedings{xfeat2024,
  title={XFeat: Accelerated Features for Lightweight Image Matching},
  author={Guilherme Potje and Felipe Cadar and Andre Araujo and Renato Martins and Erickson R. Nascimento},
  booktitle={CVPR},
  year={2024}
}
```

## License

This implementation follows the same license as the original XFeat project.
