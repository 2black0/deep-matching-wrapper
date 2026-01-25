# matcher-cpp/clidd

CLIDD inference in C++ using LibTorch (TorchScript) + OpenCV.

This is intended to be **numerically comparable** to the Python PyTorch implementation in `matcher/clidd/`.

## What runs where

- **TorchScript (LibTorch, CUDA/CPU)**
  - CLIDD forward (feature extraction + NMS + top-k selection + descriptor sampling)
  - Descriptor matching (same math as `CLIDD.match`, executed on CUDA for speed)
- **OpenCV (CPU)**
  - `cv::findHomography(..., cv::USAC_MAGSAC, ...)` for `H` and inlier mask
  - visualization output (`result.jpg`)

## Requirements

- OpenCV (your system OpenCV is fine)
- LibTorch (standalone C++ distribution)

This repo is configured to use:

- `LIBTORCH_DIR=/home/ardyseto/libtorch`

You can override it when configuring CMake.

## 1) Export TorchScript weights

LibTorch cannot directly load Python `.pth` state_dict files reliably (pickle), so we export a TorchScript `.pt` that mirrors the Python `CLIDD.forward` logic.

Command (example for `U128`, `topk=2048`):

```bash
pixi run python matcher/clidd/torchscript/convert_torchscript.py \
  --weights U128 \
  --topk 2048
```

Output:

- `matcher-cpp/clidd/weights/clidd_u128_fp32_k2048.pt`

Notes:

- TorchScript export is **FP32-only** right now.
- `--topk` is baked into the exported file name and must match what you pass to the C++ demo.

## 2) Build

Build directory is kept local to this module:

```bash
cmake -S matcher-cpp -B matcher-cpp/clidd/build \
  -DLIBTORCH_DIR=/home/ardyseto/libtorch
cmake --build matcher-cpp/clidd/build -j
```

## 3) Run demo

Example (CUDA):

```bash
./matcher-cpp/clidd/build/clidd/demo_clidd \
  --model clidd-u128 \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --topk 2048
```

To save visualization + log:

```bash
./matcher-cpp/clidd/build/clidd/demo_clidd \
  --model clidd-u128 \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --topk 2048 \
  --output yes
```

This writes:

- `outputs/matching-cpp/clidd-u128_fp32_ref_tgt/result.jpg`
- `outputs/matching-cpp/clidd-u128_fp32_ref_tgt/result.txt`

## Timing notes

The demo prints both:

- a breakdown (`infer(per-img)`, `match`, `ransac`)
- a Python-comparable `Time: ... ms` measured with a warmup call and CUDA synchronization (same semantics as `demo_matcher.py`).
