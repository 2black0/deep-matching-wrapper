# XFeat C++ Implementation

This directory contains the C++ implementation of XFeat feature matching with three variants:
- **XFeat**: Sparse features + Mutual Nearest Neighbors matching
- **XFeat-Star**: Semi-dense features + refinement
- **XFeat-LightGlue**: Sparse features + LightGlue matcher (experimental)

Based on the paper: "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024"
- Website: https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

## Directory Structure

```
matcher-cpp/xfeat/
├── CMakeLists.txt                    # Build configuration
├── include/xfeat/
│   └── XFeatTorchMatcher.h          # C++ header
├── src/
│   └── XFeatTorchMatcher.cpp         # C++ implementation
├── demo/
│   └── demo_xfeat.cpp                # Demo executable
└── weights/                          # TorchScript models (generated)
    ├── xfeat_fp32_k4096.pt           # XFeat sparse model
    └── xfeat_star_fp32_k4096.pt      # XFeat-Star model
```

## Prerequisites

1. **LibTorch** (PyTorch C++ API)
2. **OpenCV** (4.x or later)
3. **CMake** (3.16+)
4. **C++17 compatible compiler**

## Step 1: Convert Python Models to TorchScript

Before building the C++ code, you need to export the PyTorch models to TorchScript format.

### 1.1 XFeat (Sparse features)

```bash
python matcher/xfeat/torchscript/convert_torchscript_xfeat.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096 \
  --detection-threshold 0.05
```

This creates: `matcher-cpp/xfeat/weights/xfeat_fp32_k4096.pt`

### 1.2 XFeat-Star (Semi-dense)

```bash
python matcher/xfeat/torchscript/convert_torchscript_xfeat_star.py \
  --weights matcher/xfeat/weights/xfeat.pt \
  --topk 4096 \
  --fine-conf 0.25
```

This creates: `matcher-cpp/xfeat/weights/xfeat_star_fp32_k4096.pt`

### 1.3 XFeat-LightGlue (Experimental)

```bash
python matcher/xfeat/torchscript/convert_torchscript_xfeat_lightglue.py \
  --xfeat-weights matcher/xfeat/weights/xfeat.pt \
  --lightglue-weights matcher/xfeat/weights/xfeat-lighterglue.pt \
  --topk 4096
```

Note: LightGlue integration is experimental due to TorchScript limitations with kornia dependencies.

## Step 2: Build the C++ Code

```bash
cd matcher-cpp/xfeat
mkdir -p build
cd build
cmake .. -DLIBTORCH_DIR=/path/to/libtorch
make -j$(nproc)
```

This will create:
- `libdmw_xfeat_lib.so` (or `.a`) - The matcher library
- `demo_xfeat` - Demo executable

## Step 3: Run the Demo

### XFeat (Sparse + MNN)

```bash
./demo_xfeat \
  --img1 /path/to/image1.jpg \
  --img2 /path/to/image2.jpg \
  --mode xfeat \
  --device cpu \
  --topk 4096 \
  --min-cossim 0.82 \
  --output yes
```

### XFeat-Star (Semi-dense + Refinement)

```bash
./demo_xfeat \
  --img1 /path/to/image1.jpg \
  --img2 /path/to/image2.jpg \
  --mode xfeat-star \
  --device cpu \
  --topk 4096 \
  --fine-conf 0.25 \
  --output yes
```

### XFeat-LightGlue (Sparse + LightGlue - Experimental)

```bash
./demo_xfeat \
  --img1 /path/to/image1.jpg \
  --img2 /path/to/image2.jpg \
  --mode xfeat-lightglue \
  --device cpu \
  --topk 4096 \
  --min-match-conf 0.1 \
  --output yes
```

## Configuration Options

### Common Options

- `--mode`: Matching mode (`xfeat`, `xfeat-star`, `xfeat-lightglue`)
- `--device`: Device for inference (`cpu` or `cuda`)
- `--dtype`: Data type (currently only `fp32` supported)
- `--topk`: Number of top keypoints to keep (must match exported model)
- `--output`: Save output visualization (`yes` or `no`)
- `--out`: Custom output path for result image
- `--draw-all`: Draw all matches including outliers

### XFeat-specific Options

- `--detection-threshold`: Threshold for keypoint detection (default: 0.05)
- `--min-cossim`: Minimum cosine similarity for MNN matching (default: 0.82)

### XFeat-Star-specific Options

- `--fine-conf`: Confidence threshold for refinement (default: 0.25)

### XFeat-LightGlue-specific Options

- `--min-match-conf`: Minimum confidence for LightGlue matches (default: 0.1)

## Performance Comparison

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| XFeat | Fast | Good | Real-time applications |
| XFeat-Star | Medium | Better | Semi-dense correspondence |
| XFeat-LightGlue | Slower | Best | High-accuracy matching |

## Implementation Notes

1. **XFeat Mode**: Uses sparse feature extraction with mutual nearest neighbor matching
2. **XFeat-Star Mode**: Extracts features at dual scales (0.6x and 1.3x) and refines matches using an MLP
3. **XFeat-LightGlue Mode**: Currently uses sparse features with MNN as fallback (full LightGlue integration pending)

## API Usage

```cpp
#include "xfeat/XFeatTorchMatcher.h"

// Configure matcher
dmw::xfeat::XFeatConfig cfg;
cfg.mode = dmw::xfeat::XFeatMode::XFEAT;
cfg.device = "cpu";
cfg.top_k = 4096;
cfg.min_cossim = 0.82f;

// Create matcher
dmw::xfeat::XFeatTorchMatcher matcher(cfg);

// Match two images
cv::Mat img0 = cv::imread("image1.jpg");
cv::Mat img1 = cv::imread("image2.jpg");
auto result = matcher.match(img0, img1);

// Access results
std::cout << "Matches: " << result.matched_kpts0.size() << "\n";
std::cout << "Inliers: " << result.inlier_kpts0.size() << "\n";
std::cout << "Inference time: " << result.ms_infer << " ms\n";
```

## Troubleshooting

### Missing TorchScript weights error

```
Missing TorchScript weights: matcher-cpp/xfeat/weights/xfeat_fp32_k4096.pt
```

**Solution**: Run the conversion script for the appropriate mode (see Step 1).

### Different top_k value

If you exported with a different `--topk` value, make sure to use the same value when running the demo:

```bash
./demo_xfeat --topk 2048 ...  # Must match exported model
```

### CUDA errors

If running on CPU-only machine but models were exported on CUDA:
```bash
--device cpu  # Force CPU execution
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
