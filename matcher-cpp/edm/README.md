# EDM C++ Implementation

This directory contains the C++ implementation of EDM (Efficient Dense Matching) using LibTorch (TorchScript) and OpenCV.

EDM is a dense correspondence matching method that produces high-quality matches across the entire image pair.

## Directory Structure

```
matcher-cpp/edm/
├── CMakeLists.txt                    # Build configuration
├── include/edm/
│   └── EdmTorchMatcher.h            # C++ header
├── src/
│   └── EdmTorchMatcher.cpp          # C++ implementation
├── demo/
│   └── demo_edm.cpp                 # Demo executable
└── weights/                         # TorchScript models (generated)
    ├── edm_fp32_w640_h480_topk1680.pt      # EDM model (CPU)
    └── edm_fp32_w640_h480_topk1680_cuda.pt # EDM model (CUDA)
```

## Prerequisites

1. **LibTorch** (PyTorch C++ API, version 2.10.0+)
2. **OpenCV** (4.x or later)
3. **CMake** (3.16+)
4. **C++17 compatible compiler**
5. **CUDA Toolkit** (13.0+, optional for GPU acceleration)

## What Runs Where

- **TorchScript (LibTorch, CUDA/CPU)**
  - EDM forward pass (dense correspondence extraction + refinement)
- **OpenCV (CPU)**
  - `cv::findHomography(..., cv::USAC_MAGSAC, ...)` for homography estimation and inlier mask
  - Visualization output (`result.jpg`)

## Step 1: Export TorchScript Weights

LibTorch cannot directly load `.safetensors` files, so export TorchScript `.pt` models:

**Important**: EDM requires separate exports for CPU and CUDA due to device-specific optimizations.

### CPU Version

```bash
pixi run python matcher/edm/torchscript/convert_torchscript.py \
  --weights matcher/edm/weights/edm.safetensors \
  --w 640 --h 480 \
  --topk 1680 \
  --device cpu
```

Output: `matcher-cpp/edm/weights/edm_fp32_w640_h480_topk1680.pt`

### CUDA Version

```bash
pixi run python matcher/edm/torchscript/convert_torchscript.py \
  --weights matcher/edm/weights/edm.safetensors \
  --w 640 --h 480 \
  --topk 1680 \
  --device cuda
```

Output: `matcher-cpp/edm/weights/edm_fp32_w640_h480_topk1680_cuda.pt`

**Notes:**
- The C++ code automatically selects the correct model based on the `--device` parameter
- Input images are resized to `--w` × `--h` during inference
- `--topk` determines maximum number of matches to return

## Step 2: Build the C++ Code

To avoid pixi environment conflicts, build in a clean environment:

```bash
# Set environment variables
export LIBTORCH_DIR="/home/ardyseto/libtorch"
export CUDA_PATH="/usr/local/cuda-13.0"

# Clean environment build
env -i PATH="$CUDA_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake -S matcher-cpp/edm -B matcher-cpp/edm/build \
    -DLIBTORCH_DIR="$LIBTORCH_DIR" \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" \
    -DTorch_DIR="$LIBTORCH_DIR/share/cmake/Torch" \
    -DCaffe2_DIR="$LIBTORCH_DIR/share/cmake/Caffe2" \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"

# Build
env -i PATH="$CUDA_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  HOME="$HOME" \
  cmake --build matcher-cpp/edm/build -j8
```

This creates:
- `libdmw_edm_lib.a` - Static library
- `demo_edm` - Demo executable

## Step 3: Run the Demo

### CPU Execution

```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/edm/build/demo_edm \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cpu \
  --dtype fp32 \
  --w 640 --h 480 --topk 1680 \
  --output yes
```

### CUDA Execution

```bash
export LD_LIBRARY_PATH=$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

./matcher-cpp/edm/build/demo_edm \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --w 640 --h 480 --topk 1680 \
  --output yes
```

Output files (when `--output yes`):
- `outputs/matching-cpp/edm_fp32_ref_tgt/result.jpg` - Visualization
- `outputs/matching-cpp/edm_fp32_ref_tgt/result.txt` - Match statistics

## Configuration Options

- `--device`: Device for inference (`cpu` or `cuda`)
- `--dtype`: Data type (currently only `fp32` supported)
- `--w`: Target width for image resizing (must match exported model, default: 640)
- `--h`: Target height for image resizing (must match exported model, default: 480)
- `--topk`: Maximum number of matches to return (must match exported model, default: 1680)
- `--output`: Save output visualization (`yes` or `no`)
- `--out`: Custom output path for result image

## Performance Benchmarks

Test images: `assets/ref.png` (800x600) and `assets/tgt.png` (800x600)
*Note: Images are resized to 640×480 during inference*

| Device | Matches | Inliers | Inlier Ratio | Preprocess (ms) | Inference (ms) | Post (ms) | Total (ms) |
|--------|---------|---------|--------------|-----------------|----------------|-----------|------------|
| CPU | 1106 | 1101 | 99.5% | ~1 | ~262 | ~0.1 | ~263 |
| CUDA | 1107 | 1102 | 99.5% | ~2 | ~392 | ~0.1 | ~394 |

*Measured on NVIDIA RTX 4090 / AMD Ryzen 9 7950X*

**Note**: CUDA inference appears slower due to synchronization overhead on high-throughput GPUs. For batch processing or larger images, CUDA shows significant speedup.

## TorchScript Output Format

The exported TorchScript model returns a float tensor of shape `(topk, 11)` with columns:

1. `mkpts0_c.x`, `mkpts0_c.y` - Coarse matches in image 0
2. `mkpts1_c.x`, `mkpts1_c.y` - Coarse matches in image 1
3. `offset01.x`, `offset01.y` - Normalized offset from image 0 to 1
4. `offset10.x`, `offset10.y` - Normalized offset from image 1 to 0
5. `score01`, `score10` - Match scores (bidirectional)
6. `mconf` - Match confidence

The C++ implementation applies the offsets and filters by confidence to produce final matches.

## API Usage

```cpp
#include "edm/EdmTorchMatcher.h"

// Configure EDM matcher
dmw::edm::EdmConfig cfg;
cfg.device = "cuda";
cfg.dtype = "fp32";
cfg.w = 640;
cfg.h = 480;
cfg.top_k = 1680;
cfg.weights_path = "";  // Empty for auto-selection

// Create matcher
dmw::edm::EdmTorchMatcher matcher(cfg);

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
Missing TorchScript weights: matcher-cpp/edm/weights/edm_fp32_w640_h480_topk1680.pt
```

**Solution**: Export models for both CPU and CUDA (see Step 1).

### Device mismatch error

If you get device-related errors:

**Solution**: Make sure you exported both CPU and CUDA versions:
- CPU: `--device cpu` → creates `edm_fp32_w640_h480_topk1680.pt`
- CUDA: `--device cuda` → creates `edm_fp32_w640_h480_topk1680_cuda.pt`

### Different resolution/topk

If you exported with different parameters, use the same values when running:

```bash
./demo_edm --w 800 --h 600 --topk 2000 ...  # Must match exported model
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

## Implementation Notes

1. **Dense Matching**: EDM produces dense correspondences across the image pair, resulting in many high-quality matches
2. **Image Resizing**: Input images are automatically resized to the specified resolution (`w` × `h`)
3. **Device Selection**: Automatically selects the appropriate model file based on the device parameter
4. **High Inlier Ratio**: EDM typically achieves >99% inlier ratio due to dense correspondence constraints

## License

This implementation follows the same license as the original EDM project.
