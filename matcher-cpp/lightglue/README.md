# SuperPoint + LightGlue C++ Matcher

C++ implementation of SuperPoint feature detection + LightGlue feature matching using LibTorch (TorchScript).

## Overview

This matcher combines:
- **SuperPoint**: Dense keypoint detector and descriptor extractor
- **LightGlue**: Transformer-based feature matcher with attention mechanisms

**Architecture**:
1. SuperPoint extracts dense keypoint scores and descriptor maps
2. C++ applies NMS, border removal, and top-k selection to get sparse keypoints
3. C++ samples descriptors at keypoint locations using bilinear interpolation
4. LightGlue matches keypoints using transformer layers (self+cross attention)
5. C++ filters matches by mutual consistency and confidence threshold

## Model Export

Export SuperPoint and LightGlue models to TorchScript format:

```bash
# Export SuperPoint (CPU and CUDA)
pixi run python matcher/lightglue/torchscript/convert_torchscript_superpoint.py --device cpu
pixi run python matcher/lightglue/torchscript/convert_torchscript_superpoint.py --device cuda

# Export LightGlue (CPU and CUDA)
pixi run python matcher/lightglue/torchscript/convert_torchscript_lightglue.py --device cpu --max-kpts 2048
pixi run python matcher/lightglue/torchscript/convert_torchscript_lightglue.py --device cuda --max-kpts 2048
```

**Exported models**:
- `matcher-cpp/lightglue/weights/superpoint_fp32.pt` (CPU)
- `matcher-cpp/lightglue/weights/superpoint_fp32_cuda.pt` (CUDA)
- `matcher-cpp/lightglue/weights/superpoint_lightglue_fp32_k2048.pt` (CPU)
- `matcher-cpp/lightglue/weights/superpoint_lightglue_fp32_k2048_cuda.pt` (CUDA)

**Note**: Models are device-specific. The C++ matcher automatically selects the correct model based on the device parameter.

## Build

```bash
cd matcher-cpp/lightglue
mkdir -p build && cd build

# Configure (use clean environment to avoid pixi libtbb conflicts)
env -i PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake ..

# Build
env -i PATH="/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake --build . -j$(nproc)
```

## Usage

```bash
export LD_LIBRARY_PATH=/home/ardyseto/libtorch/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# CPU
./matcher-cpp/lightglue/build/demo_superpoint_lightglue \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cpu

# CUDA
./matcher-cpp/lightglue/build/demo_superpoint_lightglue \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device` | `cpu` | Device: `cpu` or `cuda` |
| `--dtype` | `fp32` | Data type (only fp32 supported) |
| `--max-kpts` | `2048` | Maximum keypoints to detect |
| `--detection-thr` | `0.005` | Keypoint detection threshold |
| `--nms-radius` | `4` | NMS radius in pixels |
| `--remove-borders` | `4` | Remove keypoints within N pixels from borders |
| `--match-thr` | `0.1` | LightGlue matching threshold (probability) |
| `--output` | `no` | Save visualization: `yes` or `no` |
| `--out` | | Custom output path (overrides `--output`) |
| `--draw-all` | | Draw all matches (not just inliers) |

## Performance (assets/ref.png + assets/tgt.png, 800x600)

### Python (Reference)
```
Device: CUDA
Keypoints: 2048 + 2048
Matches: 502
Inliers: 496 (99.0%)
Time: 56 ms
```

### C++ Implementation

**CPU**:
```
Keypoints: 2048 + 1790
Matches: 455
Inliers: 447 (98.2%)
Time: 2304 ms (warmup excluded)
  - SuperPoint0: 810 ms
  - SuperPoint1: 810 ms
  - LightGlue: 683 ms
  - RANSAC: 0.2 ms
```

**CUDA**:
```
Keypoints: 2048 + 1790
Matches: 455
Inliers: 450 (98.9%)
Time: 1747 ms (warmup excluded)
  - SuperPoint0: 415 ms
  - SuperPoint1: 411 ms
  - LightGlue: 921 ms
  - RANSAC: 0.3 ms
```

**Notes**:
- C++ is slower than Python due to disabled optimizations in TorchScript:
  - Flash attention disabled (not TorchScript-compatible)
  - Pruning/early stopping disabled (dynamic control flow)
  - No torch.compile() optimizations
- C++ extracts slightly fewer keypoints due to different NMS/threshold behavior
- Match quality is comparable (98%+ inlier ratio for both)

## Implementation Details

### Two-Stage Architecture

**Stage 1: SuperPoint Feature Extraction**
- TorchScript model outputs dense scores (H×W) and descriptor map (256×H/8×W/8)
- C++ post-processing:
  - Simple NMS with configurable radius
  - Border removal
  - Top-k selection by score
  - Bilinear descriptor sampling at keypoint locations

**Stage 2: LightGlue Matching**
- TorchScript model with transformer layers (9 layers, 4 heads)
- Outputs log-assignment matrix (N0+1 × N1+1) with dustbin row/column
- C++ post-processing:
  - Convert log-probabilities to probabilities (exp)
  - Mutual nearest neighbor matching
  - Confidence threshold filtering

### TorchScript Compatibility

LightGlue uses complex attention mechanisms that required modifications for TorchScript export:

**Disabled features**:
- Flash attention (`flash=False`)
- Dynamic pruning (`depth_confidence=-1`, `width_confidence=-1`)
- Early stopping

**Architecture preserved**:
- 9 transformer layers with self+cross attention
- Learnable positional encodings
- Matchability scores and log-assignment

### Why Separate CPU/CUDA Models?

torch.jit.trace() records operations during tracing, including device placements. Device-specific models avoid runtime device mismatch errors.

## Code Structure

```
matcher-cpp/lightglue/
├── include/lightglue/
│   └── SuperPointLightGlueMatcher.h   # Public API
├── src/
│   └── SuperPointLightGlueMatcher.cpp # Implementation
├── demo/
│   └── demo_superpoint_lightglue.cpp  # Demo executable
├── weights/                            # Exported TorchScript models
│   ├── superpoint_fp32.pt
│   ├── superpoint_fp32_cuda.pt
│   ├── superpoint_lightglue_fp32_k2048.pt
│   └── superpoint_lightglue_fp32_k2048_cuda.pt
└── CMakeLists.txt
```

## References

- **SuperPoint**: DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPRW 2018
- **LightGlue**: Lindenberger et al., "LightGlue: Local Feature Matching at Light Speed", ICCV 2023
- Original implementation: https://github.com/cvg/LightGlue
