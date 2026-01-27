# XFeat C++ Implementation

C++ implementation of XFeat feature matching with two variants:
- **XFeat**: Sparse features + Mutual Nearest Neighbors matching
- **XFeat-Star**: Semi-dense features + refinement

Based on the paper: "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024"
- Website: https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

## Prerequisites

1. **LibTorch** (PyTorch C++ API, version 2.10.0+)
2. **OpenCV** (4.x or later)
3. **CMake** (3.16+)
4. **C++17 compatible compiler**
5. **CUDA Toolkit** (13.0+, optional for GPU acceleration)

## Workflow

### 1. Export PyTorch Models to TorchScript

**XFeat (Sparse):**
```bash
pixi run python matcher/xfeat/torchscript/convert_torchscript_xfeat.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096 \
  --detection-threshold 0.05
```

**XFeat-Star (Semi-dense) - CPU:**
```bash
pixi run python matcher/xfeat/torchscript/convert_torchscript_xfeat_star.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096 \
  --fine-conf 0.25 \
  --device cpu
```

**XFeat-Star (Semi-dense) - CUDA:**
```bash
pixi run python matcher/xfeat/torchscript/convert_torchscript_xfeat_star.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096 \
  --fine-conf 0.25 \
  --device cuda
```

This creates TorchScript models in `matcher-cpp/xfeat/weights/`.

### 2. Build

```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-13.0

cd matcher-cpp/xfeat
rm -rf build && mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH=/home/ardyseto/libtorch \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
      ..

make -j$(nproc)
```

### 3. Run Demo

**XFeat on CPU:**
```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH

./matcher-cpp/xfeat/build/demo_xfeat \
  --mode xfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cpu \
  --output yes
```

**XFeat on CUDA:**
```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH

./matcher-cpp/xfeat/build/demo_xfeat \
  --mode xfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --output yes
```

**XFeat-Star on CPU:**
```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH

./matcher-cpp/xfeat/build/demo_xfeat \
  --mode xfeat-star \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cpu \
  --output yes
```

**XFeat-Star on CUDA:**
```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:$LD_LIBRARY_PATH

./matcher-cpp/xfeat/build/demo_xfeat \
  --mode xfeat-star \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --output yes
```

## Options

### Common Options
- `--mode`: Matching mode (`xfeat` or `xfeat-star`)
- `--device`: Device (`cpu` or `cuda`)
- `--dtype`: Data type (`fp32` only)
- `--topk`: Number of top keypoints (default: 4096, must match exported model)
- `--output`: Save visualization (`yes` or `no`)
- `--out`: Custom output path
- `--draw-all`: Draw outliers too

### XFeat-specific
- `--detection-threshold`: Keypoint detection threshold (default: 0.05)
- `--min-cossim`: Minimum cosine similarity for MNN matching (default: -1, disabled)

### XFeat-Star-specific
- `--fine-conf`: Confidence threshold for refinement (default: 0.25)

## Performance Benchmarks

Test images: `assets/ref.png` and `assets/tgt.png` (800x600)

### XFeat (Sparse)
| Device | Matches | Inliers | Ratio | Time |
|--------|---------|---------|-------|------|
| CPU | 832 | 157 | 18.9% | 366ms |
| CUDA | 832 | 154 | 18.5% | 315ms |

### XFeat-Star (Semi-dense)
| Device | Matches | Inliers | Ratio | Time |
|--------|---------|---------|-------|------|
| CPU | 653 | 347 | 53.1% | 101ms |
| CUDA | 650 | 349 | 53.7% | 117ms |

*Measured on NVIDIA RTX 4060 Ti / AMD Ryzen*

## API Usage

```cpp
#include "xfeat/XFeatTorchMatcher.h"

// Configure matcher
dmw::xfeat::XFeatConfig cfg;
cfg.mode = dmw::xfeat::XFeatMode::XFEAT;
cfg.device = "cpu";
cfg.top_k = 4096;
cfg.min_cossim = -1.0f;

// Create matcher
dmw::xfeat::XFeatTorchMatcher matcher(cfg);

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
Missing TorchScript weights: matcher-cpp/xfeat/weights/xfeat_fp32_k4096.pt
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

## Citation

```bibtex
@inproceedings{xfeat2024,
  title={XFeat: Accelerated Features for Lightweight Image Matching},
  author={Guilherme Potje and Felipe Cadar and Andre Araujo and Renato Martins and Erickson R. Nascimento},
  booktitle={CVPR},
  year={2024}
}
```
