# CLIDD-TensorRT

High-performance C++ and Python feature matching pipeline using CLIDD with TensorRT acceleration.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

## Overview

CLIDD-TensorRT provides fast and accurate feature matching for visual odometry and SLAM systems:

- **High Performance**: C++ inference with TensorRT GPU acceleration
- **Single Forward Pass**: Unlike LiftFeat, CLIDD uses one unified model
- **Multiple Precision**: FP32, FP16, and INT8 engine support
- **Production Ready**: Optimized models for multiple resolutions
- **Geometric Validation**: RANSAC-based inlier filtering with homography
- **Visualization**: Built-in match visualization tools

### Architecture

```
Input Images → Preprocess (pad to 32) → TensorRT Engine → Matching → RANSAC → Output
```

Unlike LiftFeat-TensorRT which uses two forward passes (TensorRT + TorchScript), CLIDD is fully contained in a single TensorRT engine.

## Quick Start

### Prerequisites

- CUDA 12.0+ with cuDNN
- TensorRT 10.0+
- LibTorch 2.0+ (CUDA)
- OpenCV 4.0+ with CUDA support
- CMake 3.16+
- GCC 12.0+ with C++17
- Python 3.10+ with pixi (for model export)

### Build

```bash
cd matcher-tensorrt/CLIDD-TensorRT
bash scripts/build.sh
```

### Test

```bash
./build/test_inference_cpp \
  --engine weights/clidd_m64_640x480_fp16.engine \
  --img1 test/assets/ref.png \
  --img2 test/assets/tgt.png \
  --loop 10 \
  --inliers-only
```

## Project Structure

```
.
├── src/                              # Core library
│   ├── clidd_trt.cpp                # Main implementation
│   └── clidd_trt.cpp               # Public header
├── test/                             # Tests and benchmarks
│   ├── test_inference.cpp           # C++ inference test
│   └── assets/                      # Test images
│       ├── ref.png
│       └── tgt.png
├── scripts/                          # Utility scripts
│   ├── build_all_engines.py        # Build FP32/FP16/INT8 engines
│   └── build.sh                    # Main build script
├── weights/                         # Pre-built models
│   ├── clidd_m64_*.engine          # TensorRT engines
├── CMakeLists.txt                  # Build configuration
└── README.md                        # This file
```

## Building Engines

### 1. Build ONNX and TensorRT Engine (M64 variant)

```bash
cd matcher-tensorrt/CLIDD-TensorRT
python3 scripts/build_all_engines.py \
  --variant M64 \
  --resolution 640x480 \
  --precision fp16 \
  --topk 2048
```

### 2. Build All Resolutions

```bash
python3 scripts/build_all_engines.py \
  --variant M64 \
  --all-res \
  --precision fp16
```

### 3. Available Variants

| Variant | Descriptor Dim | Description |
|---------|----------------|-------------|
| A48     | 48             | Lightest model |
| M64     | 64             | Medium (default) |
| U128    | 128            | Largest, highest accuracy |

### 4. Build Specific Engine Manually

#### Export ONNX

```bash
python3 -c "
import torch
import sys
sys.path.insert(0, '/home/ardyseto/Documents/GitHub/deep-matching-wrapper')

from matcher.clidd.modules.model import Model
from matcher.clidd.modules.clidd_wrapper import CLIDD

cfg_params = CLIDD.cfgs['M64']
model = Model(**cfg_params)
state = torch.load('/home/ardyseto/Documents/GitHub/deep-matching-wrapper/matcher/clidd/weights/M64.pth', map_location='cpu')
model.load_state_dict(state)
model.eval()

# Export with full post-processing
class CLIDDExport(torch.nn.Module):
    def __init__(self, model, top_k=2048, radius=2, score_thresh=-5.0, border=4):
        super().__init__()
        self.model = model
        self.top_k = top_k
        self.radius = radius
        self.score_thresh = score_thresh
        self.border = border
        self.mp = torch.nn.MaxPool2d(radius * 2 + 1, 1, radius)
    
    def forward(self, x):
        B, C, H, W = x.shape
        size = torch.tensor([W, H], dtype=x.dtype, device=x.device)
        
        raw_desc, raw_detect = self.model(x)
        
        is_max = raw_detect == self.mp(raw_detect)
        
        if self.border > 0:
            b = self.border
            border_mask = torch.ones_like(is_max, dtype=torch.bool)
            border_mask[..., :, :b] = False
            border_mask[..., :, -b:] = False
            border_mask[..., :b, :] = False
            border_mask[..., -b:, :] = False
            is_max = is_max & border_mask
        
        is_good = raw_detect > self.score_thresh
        valid = (is_max & is_good)
        
        neg_inf = torch.full_like(raw_detect, -1e8)
        refined = torch.where(valid, raw_detect, neg_inf)
        
        flat_scores = refined.view(B, -1)
        scores, indices = torch.topk(flat_scores, k=self.top_k, dim=1)
        
        y = (indices // W).to(torch.float32)
        x_coords = (indices % W).to(torch.float32)
        kpts = torch.stack([x_coords, y], dim=-1)
        
        norm_kpts = (kpts + 0.5) / size.to(torch.float32) * 2 - 1
        norm_kpts = norm_kpts.unsqueeze(2).to(x.dtype)
        
        descriptors = self.model.sample(list(raw_desc), norm_kpts)
        
        return kpts.to(torch.float32), scores.to(torch.float32), descriptors.to(torch.float32)

exported = CLIDDExport(model).eval()
x = torch.randn(1, 3, 640, 480)

torch.onnx.export(
    exported,
    x,
    'weights/clidd_m64_640x480.onnx',
    input_names=['image'],
    output_names=['keypoints', 'scores', 'descriptors'],
    opset_version=18,
)
"
```

#### Build TensorRT Engine

```bash
# FP32
/home/ardyseto/tensorrt/bin/trtexec \
  --onnx=weights/clidd_m64_640x480.onnx \
  --saveEngine=weights/clidd_m64_640x480_fp32.engine \
  --memPoolSize=workspace:4096M

# FP16
/home/ardyseto/tensorrt/bin/trtexec \
  --onnx=weights/clidd_m64_640x480.onnx \
  --saveEngine=weights/clidd_m64_640x480_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096M
```

## Building the C++ Project

### Using build.sh

```bash
cd matcher-tensorrt/CLIDD-TensorRT
bash scripts/build.sh
```

### Manual Build

```bash
cd matcher-tensorrt/CLIDD-TensorRT
rm -rf build && mkdir build

cmake .. \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBTORCH_ROOT=/home/ardyseto/libtorch \
  -DTENSORRT_ROOT=/home/ardyseto/tensorrt

cmake --build build -j$(nproc)
```

## Testing

### C++ Test

```bash
# Basic test
./build/test_inference_cpp \
  --engine weights/clidd_m64_640x480_fp16.engine \
  --img1 test/assets/ref.png \
  --img2 test/assets/tgt.png

# Benchmark with 10 loops
./build/test_inference_cpp \
  --engine weights/clidd_m64_640x480_fp16.engine \
  --img1 test/assets/ref.png \
  --img2 test/assets/tgt.png \
  --loop 10

# Show only inliers
./build/test_inference_cpp \
  --engine weights/clidd_m64_640x480_fp16.engine \
  --img1 test/assets/ref.png \
  --img2 test/assets/tgt.png \
  --loop 10 \
  --inliers-only

# Custom parameters
./build/test_inference_cpp \
  --engine weights/clidd_m64_640x480_fp16.engine \
  --top-k 1024 \
  --score-thresh -3.0 \
  --beta 30 \
  --img1 test/assets/ref.png \
  --img2 test/assets/tgt.png
```

## Command-Line Interface

### C++ Test Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--img1` | test/assets/ref.png | First image |
| `--img2` | test/assets/tgt.png | Second image |
| `--engine` | (required) | TensorRT engine |
| `--top-k` | 2048 | Max keypoints |
| `--score-thresh` | -5.0 | Detection score threshold |
| `--radius` | 2 | NMS radius |
| `--border` | 4 | Border suppression |
| `--match-thresh` | 0.01 | Match threshold |
| `--beta` | 20.0 | Matching beta |
| `--out` | matches_visualization_cpp.png | Output path |
| `--loop` | 1 | Benchmark loops |
| `--inliers-only` | false | Draw only inliers |
| `--help` | - | Help |

## Key Implementation Details

### Preprocessing
Images padded to multiples of 32 (matching Python):
```python
_H = ceil(H / 32) * 32
_W = ceil(W / 32) * 32
```

### Engine Naming
`clidd_m64_{width}x{height}.engine`:
- 640×480 image → padded 640×480 → engine `clidd_m64_640x480_*.engine`

### Matching Algorithm
CLIDD uses soft matching with exponential transformation:
```python
sim = exp((dot - 1) * beta)
sim = sim^2 / (sum1 * sum2)
```

### RANSAC
Uses OpenCV findHomography with RANSAC:
- Threshold: 3.0 pixels
- Confidence: 0.9999
- Max iterations: 2000

## CLIDD vs LiftFeat-TensorRT

| Aspect | CLIDD-TensorRT | LiftFeat-TensorRT |
|--------|----------------|-------------------|
| Forward Passes | 1 | 2 |
| TensorRT | Yes | Yes |
| TorchScript | No | Yes (forward2) |
| Complexity | Lower | Higher |
| Model Variants | A48, M64, U128 | Single |

## Troubleshooting

### Engine Not Found
```bash
ls -lah weights/*.engine
```

### Build Errors
Verify paths in CMakeLists.txt: LIBTORCH_ROOT, TENSORRT_ROOT

### Slow First Inference
Normal - TensorRT JIT compiles on first run. Use `--loop 50`+ for steady-state.

### No Matches Found
Try adjusting:
- `--score-thresh` (lower = more keypoints)
- `--match-thresh` (lower = more matches)
- `--beta` (higher = sharper matching)

## License

MIT License - see [LICENSE](LICENSE)

## References

- [CLIDD Paper](https://github.com/zpwang-lab/CLIDD)
- [TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Original CLIDD Implementation](https://github.com/zpwang-lab/CLIDD)
