# Deep Matching Wrapper

A unified, minimal-dependency wrapper for state-of-the-art deep learning image matching algorithms. This project provides clean, standalone implementations of various feature matchers with a consistent API.

## 🎯 Supported Matchers

### Dense Matchers (Detector-Free)
- **EfficientLoFTR** (`eloftr`) - Efficient Local Feature Transformer for dense matching
- **EDM** (`edm`) - Explicit Dense Matching
- **LiftFeat** (`liftfeat`) - Feature boosting wrapper

### Sparse Matchers (Feature-Based)
- **XFeat** (`xfeat`, `xfeat-star`, `xfeat-lightglue`) - Fast and accurate sparse matching
- **SuperPoint + LightGlue** (`superpoint-lightglue`) - Classical combination with LightGlue
- **GIM** (`gim-lightglue`) - SuperPoint+LightGlue finetuned on 100h of data
- **CLIDD** (`clidd-a48`, `clidd-n64`, `clidd-t64`, `clidd-s64`, `clidd-m64`, `clidd-l64`, `clidd-g128`, `clidd-e128`, `clidd-u128`) - Lightweight Feature Matching (9 variants)

### Subpixel Refinement
- **Keypt2Subpx** (`xfeat-subpx`, `xfeat-lightglue-subpx`, `superpoint-lightglue-subpx`) - Keypoint to subpixel refinement

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

### Basic Testing

Test a single matcher:
```bash
pixi run python test_matcher.py --matcher xfeat
```

Test with custom images:
```bash
pixi run python test_matcher.py --matcher eloftr --img1 path/to/img1.png --img2 path/to/img2.png
```

Test all matchers:
```bash
pixi run python test_matcher.py --matcher all
```

### Available Arguments

```
--matcher MATCHER   Matcher name (see supported matchers above) or 'all'
--img1 IMG1        Path to first image (default: assets/ref.png)
--img2 IMG2        Path to second image (default: assets/tgt.png)
```

## 📊 Example Results

Using `assets/ref.png` and `assets/tgt.png`:

| Matcher | Matches | Inliers | Time (s) | Accuracy |
|---------|---------|---------|----------|----------|
| eloftr | 1145 | 1124 | 0.098 | 98.2% |
| edm | 1790 | 1781 | 0.036 | 99.5% |
| clidd-u128 | 865 | 666 | 0.014 | 77.0% |
| superpoint-lightglue-subpx | 502 | 480 | 0.068 | 95.6% |
| xfeat | 830 | 143 | 0.013 | 17.2% |

## 🏗️ Project Structure

```
deep-matching-wrapper/
├── matcher/                 # Matcher implementations
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
├── test_matcher.py         # Main testing script
├── assets/                 # Test images
├── scripts/                # Utility scripts
└── pixi.toml              # Pixi dependency configuration
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
