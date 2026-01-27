# matcher-cpp/edm

EDM inference in C++ using LibTorch (TorchScript) + OpenCV.

This is intended to be **numerically comparable** to the Python PyTorch EDM deploy path in `matcher/edm/modules/edm.py`.

## What runs where

- **TorchScript (LibTorch, CUDA/CPU)**
  - EDM forward (deploy-mode) from a TorchScript export
- **OpenCV (CPU)**
  - `cv::findHomography(..., cv::USAC_MAGSAC, ...)` for `H` and inlier mask
  - visualization output (`result.jpg`)

## 1) Export TorchScript weights

LibTorch cannot consume `.safetensors` directly, so we export a TorchScript `.pt` from `matcher/edm/weights/edm.safetensors`.

Command (fixed-shape export used by the C++ module):

```bash
pixi run python matcher/edm/torchscript/convert_torchscript.py \
  --weights matcher/edm/weights/edm.safetensors \
  --w 640 --h 480 \
  --topk 1680
```

Output:

- `matcher-cpp/edm/weights/edm_fp32_w640_h480_topk1680.pt`

## 2) Build

```bash
cmake -S matcher-cpp -B matcher-cpp/edm/build \
  -DLIBTORCH_DIR=/home/ardyseto/libtorch
cmake --build matcher-cpp/edm/build -j
```

## 3) Run demo

```bash
./matcher-cpp/edm/build/edm/demo_edm \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --device cuda \
  --dtype fp32 \
  --w 640 --h 480 --topk 1680 \
  --output yes
```

This writes:

- `outputs/matching-cpp/edm_fp32_ref_tgt/result.jpg`
- `outputs/matching-cpp/edm_fp32_ref_tgt/result.txt`

## TorchScript output format

The exported TorchScript returns a float tensor `(topk, 11)` with columns:

1. `mkpts0_c.x`, `mkpts0_c.y`
2. `mkpts1_c.x`, `mkpts1_c.y`
3. `offset01.x`, `offset01.y` (normalized; multiplied by `local_resolution` in C++)
4. `offset10.x`, `offset10.y` (normalized)
5. `score01`, `score10`
6. `mconf`
