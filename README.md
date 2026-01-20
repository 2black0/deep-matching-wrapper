# Deep Matching Wrapper

A unified, minimal-dependency wrapper for state-of-the-art deep learning image matching algorithms. This project provides clean, standalone implementations of various feature matchers with a consistent API.

## 🎯 Supported Matchers

### Dense Matchers (Detector-Free)

- **EfficientLoFTR** (`eloftr`) - Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed
- **EDM** (`edm`) - Efficient Deep Feature Matching
- **LiftFeat** (`liftfeat`) - 3D Geometry-Aware Local Feature Matching

### Sparse Matchers (Feature-Based)

- **XFeat** (`xfeat`, `xfeat-star`, `xfeat-lightglue`) - Accelerated Features for Lightweight Image Matching
- **SuperPoint + LightGlue** (`superpoint-lightglue`) - Self-Supervised Interest Point Detection and Description with Local Feature Matching at Light Speed
- **GIM** (`gim-lightglue`) - Learning Generalizable Image Matcher From Internet Videos
- **CLIDD** (`clidd-a48`, `clidd-n64`, `clidd-t64`, `clidd-s64`, `clidd-m64`, `clidd-l64`, `clidd-g128`, `clidd-e128`, `clidd-u128`) - Cross-Layer Independent Deformable Description for Efficient and Discriminative Local Feature Representation (9 variants)

### Subpixel Refinement

- **Keypt2Subpx** (`xfeat-subpx`, `xfeat-lightglue-subpx`, `superpoint-lightglue-subpx`) - Learning to Make Keypoints Sub-Pixel Accurate

### Handcrafted Features

- **ORB** (`orb-nn`) - ORB features with nearest neighbor matching
- **SIFT** (`sift-nn`, `sift-lightglue`) - SIFT features with NN or LightGlue matching

**Total: 23 matchers** supported!

## 📦 Installation

### 1. Install Dependencies

```bash
pixi install
```

### 2. OpenCV Setup

**Option A: Use Custom OpenCV Build (with CUDA)**

If you've built OpenCV with CUDA support:

```bash
pixi run link-opencv
```

This will symlink your system OpenCV to the pixi environment.

**Option B: Use Pixi's OpenCV**

Add to `pixi.toml`:

```toml
[dependencies]
opencv = "*"
```

Then run:

```bash
pixi install
```

### 3. Verify Installation

```bash
pixi run bash scripts/check.sh
```

This will check:

- Python environment
- PyTorch CUDA availability
- OpenCV installation
- GPU devices

## 🚀 Usage

### Single Image Matching

Test a single matcher:

```bash
pixi run python demo_matcher.py --matcher xfeat
```

Test with custom images:

```bash
pixi run python demo_matcher.py --matcher eloftr --img1 path/to/img1.png --img2 path/to/img2.png
```

Test all matchers:

```bash
pixi run python demo_matcher.py --matcher all
```

#### Available Arguments

```
--matcher MATCHER  Matcher name (xfeat, xfeat-star, xfeat-lightglue, liftfeat, gim-lightglue, edm, orb-nn, sift-nn, sift-lightglue, superpoint-lightglue, xfeat-subpx, xfeat-lightglue-subpx, superpoint-lightglue-subpx, clidd-a48, clidd-n64, clidd-t64, clidd-s64, clidd-m64, clidd-l64, clidd-g128, clidd-e128, clidd-u128, eloftr) or 'all'
--img1 IMG1        Path to first image (default: assets/ref.png)
--img2 IMG2        Path to second image (default: assets/tgt.png)
```

#### 📊 Example Results

Using `assets/ref.png` and `assets/tgt.png`:

| Model                          | Matches | Inliers | Time (s) |  Ratio   |
| :----------------------------- | :-----: | :-----: | :------: | :------: |
| **edm**                        |  1790   |  1781   |  0.035   |  99.5%   |
| **eloftr**                     |  1145   |  1124   |  0.100   |  98.2%   |
| **xfeat-lightglue-subpx**      |   780   |   707   |  0.034   |  90.6%   |
| **xfeat-lightglue**            |   780   |   703   |  0.016   |  90.1%   |
| **clidd-u128**                 |   865   |   666   |  0.014   |  77.0%   |
| **clidd-e128**                 |   806   |   580   |  0.011   |  72.0%   |
| **gim-lightglue**              |   564   |   559   |  0.093   |  99.1%   |
| **clidd-g128**                 |   741   |   508   |  0.011   |  68.6%   |
| **superpoint-lightglue-subpx** |   502   |   480   |  0.071   |  95.6%   |
| **superpoint-lightglue**       |   502   |   474   |  0.057   |  94.4%   |
| **xfeat-star**                 |   615   |   454   |  0.018   |  73.8%   |
| **clidd-l64**                  |   846   |   424   |  0.006   |  50.1%   |
| **clidd-m64**                  |   801   |   336   |  0.006   |  41.9%   |
| **liftfeat**                   |   778   |   278   |  0.037   |  35.7%   |
| **clidd-s64**                  |   744   |   231   |  0.005   |  31.0%   |
| **clidd-n64**                  |   720   |   194   |  0.004   |  26.9%   |
| **orb-nn**                     |   600   |   164   |  0.023   |  27.3%   |
| **clidd-t64**                  |   708   |   164   |  0.005   |  23.2%   |
| **xfeat-subpx**                |   830   |   150   |  0.031   |  18.1%   |
| **xfeat**                      |   830   |   143   |  0.012   |  17.2%   |
| **sift-nn**                    |   507   |   85    |  0.089   |  16.8%   |
| **sift-lightglue**             |   110   |   48    |  0.137   |  43.6%   |
| **clidd-a48**                  |   762   |   23    |  0.007   |   3.0%   |

> **Note:** Tests were run on CUDA with NVIDIA GeForce RTX 4060 Ti. Sorted by Inliers count.

Benchmark on HPatches and MegaDepth-1500 dataset can be found detailed in [HPatches Result](docs/HPATCH_RESULT.md) and [MegaDepth Result](docs/MEGADEPTH_RESULT.md).

### Real-time Matching

Run real-time matching using your webcam:

```bash
pixi run python demo_realtime.py --matcher xfeat --cam 0
```

## 🏗️ Project Structure

```
deep-matching-wrapper/
├── matcher/                # Matcher implementations
│   ├── base_matcher.py     # Base class with unified API
│   ├── xfeat/              # XFeat wrapper
│   ├── liftfeat/           # LiftFeat wrapper
│   ├── gim/                # GIM wrapper
│   ├── lightglue/          # SuperPoint+LightGlue wrapper
│   ├── edm/                # EDM wrapper
│   ├── clidd/              # CLIDD wrapper (9 variants)
│   ├── eloftr/             # EfficientLoFTR wrapper
│   ├── subpx/              # Subpixel refinement wrapper
│   └── handcrafted.py      # ORB/SIFT wrapper
├── demo_matcher.py         # Main demo script
├── check_matcher.py        # Check supported matchers script
├── demo_realtime.py        # Real-time matcher demo
├── assets/                 # Test images
├── scripts/                # Utility scripts
└── pixi.toml               # Pixi dependency configuration
```

## 🙏 Credits

This project builds upon excellent work from the computer vision research community:

### Original Implementations & Models

- **[image-matching-models (IMM)](https://github.com/gmberton/image-matching-models)** - Comprehensive image matching benchmark (original inspiration)
- **[XFeat](https://github.com/verlab/accelerated_features)** - Accelerated Features for Lightweight Image Matching
- **[LightGlue](https://github.com/cvg/LightGlue)** - Local Feature Matching at Light Speed
- **[LiftFeat](https://github.com/ShngJZ/LiftFeat)** - Learning Feature Matching via Differentiable Pose Estimation
- **[GIM](https://github.com/xuelunshen/gim)** - Generalist Foundation Model for Image Matching
- **[EDM](https://github.com/zpwang-lab/EDM)** - Explicit Dense Matching
- **[CLIDD](https://github.com/zpwang-lab/CLIDD)** - Compact Lightweight Image Descriptor and Detector
- **[EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR)** - Efficient Local Feature Transformer
- **[Keypt2Subpx](https://github.com/KimSinjeong/keypt2subpx)** - From Keypoints to Subpixel

### Related Projects

- **[SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)** - Self-Supervised Interest Point Detection and Description
- **[OpenCV](https://opencv.org/)** - Computer Vision library (ORB, SIFT implementations)

## 📝 License

This project wraps various open-source implementations. Please refer to each original repository for their respective licenses.

## 🔧 Development

Built with:

- **[Pixi](https://pixi.sh/)** - Fast, reproducible development environment
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **Python 3.10+**

---

**Note**: All matchers have been adapted to use minimal dependencies and a unified API for ease of use. Weights are automatically downloaded on first use or can be manually placed in `matcher/*/weights/` directories.
