# matcher-cpp/liftfeat

LiftFeat inference in C++ using LibTorch (TorchScript) + OpenCV.

This is intended to be **numerically comparable** to the Python PyTorch implementation in `matcher/liftfeat/`.

## What runs where

- **TorchScript (LibTorch, CUDA/CPU)**
  - LiftFeat forward (feature extraction + NMS + top-k selection + descriptor sampling)
  - Mutual nearest-neighbor descriptor matching (cosine similarity)
- **OpenCV (CPU)**
  - `cv::findHomography(..., cv::USAC_MAGSAC, ...)` for `H` and inlier mask
  - visualization output (`result.jpg`)

## Requirements

- OpenCV (your system OpenCV is fine)
- LibTorch (standalone C++ distribution)

This repo is configured to use:

- `LIBTORCH_DIR=/home/ardyseto/libtorch`

## 1) Export TorchScript weights

LibTorch cannot directly load Python `.pth` state_dict files reliably (pickle), so we export a TorchScript `.pt` that mirrors the Python `LiftFeat.extract` logic.

Command (example `topk=4096`):

```bash
pixi run python matcher/liftfeat/torchscript/convert_torchscript.py \
  --topk 4096 \
  --detect-threshold 0.05
```

Output:

- `matcher-cpp/liftfeat/weights/liftfeat_fp32_k4096.pt`

Notes:

- TorchScript export is **FP32-only** right now.
- `--topk` is baked into the exported file name and must match what you pass to the C++ demo.

## 2) Build

```bash
cmake -S matcher-cpp -B matcher-cpp/liftfeat/build \
  -DLIBTORCH_DIR=/home/ardyseto/libtorch
cmake --build matcher-cpp/liftfeat/build -j
```

## 3) Run demo

Example (CUDA):

```bash
./matcher-cpp/liftfeat/build/liftfeat/demo_liftfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --topk 4096
```

## Parity notes (Python vs C++)

With the default settings above (`topk=4096`, `detect-threshold=0.05`), the C++ outputs should be very close to Python on the included sample pair:

```bash
pixi run python demo_matcher.py --matcher liftfeat --img1 assets/ref.png --img2 assets/tgt.png
./matcher-cpp/liftfeat/build/liftfeat/demo_liftfeat --img1 assets/ref.png --img2 assets/tgt.png --device cuda --dtype fp32 --topk 4096
```

You should see nearly identical:

- total keypoints (`4096` per image)
- matched keypoints (typically off by ~0-2)
- inliers (typically off by ~0-2)

To save visualization + log:

```bash
./matcher-cpp/liftfeat/build/liftfeat/demo_liftfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --topk 4096 \
  --output yes
```

This writes:

- `outputs/matching-cpp/liftfeat_fp32_ref_tgt/result.jpg`
- `outputs/matching-cpp/liftfeat_fp32_ref_tgt/result.txt`
