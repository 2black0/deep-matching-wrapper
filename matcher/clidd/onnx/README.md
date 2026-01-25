# CLIDD ONNX Export

This folder contains scripts to export CLIDD models to ONNX, validate numerical parity against the PyTorch implementation, and run an end-to-end ONNX matching demo.

Files:
- `matcher/clidd/onnx/convert_onnx.py`: export a chosen CLIDD configuration (e.g. `A48`) to ONNX (FP32/FP16)
- `matcher/clidd/onnx/check_onnx.py`: compare PyTorch vs ONNX outputs (keypoints, scores, descriptors)
- `matcher/clidd/onnx/demo_onnx.py`: run detection + description + matching using ONNX Runtime

Exported models are written to `matcher/clidd/weights/`.

## Background: CLIDD architecture in this repo

The CLIDD implementation is split across:

- `matcher/clidd/modules/model.py`:
  - feature extraction backbone (multi-scale pyramid)
  - detection head producing a dense score map
  - descriptor sampling via a deformable sampling + projection operator
- `matcher/clidd/modules/triton_plugin.py`:
  - `deformable_sample_project(...)` implemented in pure PyTorch using `grid_sample` + `einsum`
  - this replaces the original Triton kernel so export/CPU works
- `matcher/clidd/modules/clidd_wrapper.py`:
  - inference wrapper that handles resizing, NMS, top-k selection, descriptor sampling, and matching

### What the model produces

Given an input image `x` of shape `(B, 3, H, W)`:

1) `Model.forward(x)` returns:
- `raw_desc`: a tuple of three feature maps `(x1, x2, x3)`
- `raw_detect`: a dense detection score map of shape `(B, 1, H, W)` (after PixelShuffle upsampling)

2) `Model.sample(raw_desc, kpts_norm)` returns:
- descriptors at sparse keypoints, shape `(B, N, Cdesc)`
- descriptors are L2-normalized (`F.normalize(..., 2, -1)`)

3) `CLIDD.forward(x)` (wrapper) performs:
- resize to multiples of 32 (bilinear, `align_corners=True`)
- NMS maxima via maxpool
- border suppression of 4px
- score thresholding (default `score=-5`)
- top-k selection
- descriptor sampling using `(kpts + 0.5) / size * 2 - 1` normalized coordinates

## Export format

Unlike some pipelines that export intermediate dense maps, this exporter produces the wrapper-level sparse outputs directly:

- `keypoints`: `(1, topk, 2)` in pixel coordinates `(x, y)`
- `scores`: `(1, topk)` detection scores
- `descriptors`: `(1, topk, Cdesc)` L2-normalized descriptors

This matches the practical downstream usage: ONNX outputs are ready for matching.

## 1) Convert / Export

From the repository root:

```bash
pixi run python matcher/clidd/onnx/convert_onnx.py \
  --weights A48 \
  --size 640 480 \
  --dtype FP32 \
  --topk 1024

pixi run python matcher/clidd/onnx/convert_onnx.py \
  --weights A48 \
  --size 640 480 \
  --dtype FP16 \
  --topk 1024
```

Outputs:
- `matcher/clidd/weights/clidd_a48_fp32_640x480.onnx`
- `matcher/clidd/weights/clidd_a48_fp16_640x480.onnx`

You may also see external tensor data files:
- `matcher/clidd/weights/*.onnx.data`
Keep them next to the `.onnx` file.

### Exporter implementation details (academic notes)

The exporter is implemented as `CLIDDExport` in `matcher/clidd/onnx/convert_onnx.py`.
It wraps `Model(...)` and reproduces the *same inference-time post-processing* as `matcher/clidd/modules/clidd_wrapper.py`.

This is crucial: exporting only the raw network outputs (`raw_desc`, `raw_detect`) is not enough to reproduce the end-to-end behavior, because CLIDD is defined by the combination of:
- detection map + NMS policy
- thresholding policy
- top-k selection
- descriptor sampling at the selected keypoints

Key steps in the exporter forward pass:

1) Backbone inference
- The exporter calls `raw_desc, raw_detect = self.model(x)`.
- `raw_desc` is a tuple of feature maps; `raw_detect` is the dense score map.

2) NMS maxima
- Uses `MaxPool2d(kernel=2*radius+1, stride=1, padding=radius)`.
- Computes `is_max = raw_detect == maxpool(raw_detect)`.

3) Border suppression
- CLIDD suppresses a 4-pixel border in the wrapper to reduce unstable detections.
- The exporter reproduces this by masking the first/last `border` pixels.

4) Score thresholding
- Wrapper uses `raw_detect > score_thresh` where default `score_thresh = -5`.
- Export uses the same.

5) **Top-k selection correctness note**

It is tempting to do:

`refined = raw_detect * mask`

This is wrong for CLIDD because `raw_detect` may contain negative values.
Multiplying non-max positions by zero sets them to `0`, which can become *larger* than real negative maxima, causing `topk` to select invalid points.

Correct approach:
- use `torch.where(valid, raw_detect, -1e8)` (a very negative sentinel)
- then apply `topk` on the flattened score map

6) Keypoint coordinate generation
- Indices from `topk` are mapped back to `(x, y)` via:
  - `y = indices // W`
  - `x = indices % W`

7) Descriptor sampling
- Normalized coordinates match the wrapper:
  - `kpts_norm = (kpts + 0.5) / size * 2 - 1`
- The exporter calls:
  - `descriptors = self.model.sample(list(raw_desc), kpts_norm.unsqueeze(2))`

8) Output types
- Export returns `keypoints`, `scores`, `descriptors` as float32 for downstream convenience.

### FP16 notes

FP16 export is supported, but simplification/optimization can sometimes change outputs (especially keypoint selection) due to precision effects.
For academic evaluation:
- prefer FP32 for parity experiments
- if you use FP16, validate it with `check_onnx.py` and treat `onnxsim` warnings seriously

## 2) Numerical check (PyTorch vs ONNX)

Run:

```bash
pixi run python matcher/clidd/onnx/check_onnx.py --weights A48 --topk 1024
```

What it does:
- Runs the **real** PyTorch wrapper baseline (`matcher/clidd/modules/clidd_wrapper.py:CLIDD`).
- Runs ONNX Runtime inference for FP32 and FP16 exports.
- Compares:
  - keypoint spatial precision/mean distance
  - scores and descriptors after spatial nearest-neighbor alignment (so re-ordering does not dominate MSE)

Interpretation guidelines:
- FP32 should be near-identical.
- FP16 typically has small drift; the key metric is that spatial precision remains high and descriptor drift remains low.

## 3) ONNX demo (end-to-end matching)

Run:

```bash
pixi run python matcher/clidd/onnx/demo_onnx.py \
  --weights A48 \
  --img1 assets/ref.png \
  --img2 assets/tgt.png
```

This demo:
- runs the exported model on both images
- matches using a NumPy port of `CLIDD.match(...)`
- estimates inliers via RANSAC homography
- saves `matcher/clidd/onnx/result.jpg`

By default the visualization draws only inliers (cleaner). Use `--draw-all` to draw all matches.

Useful parameters:
- `--dtype fp32|fp16`: choose which ONNX file to load
- `--beta`: matching sharpness (default `20` like wrapper)
- `--min-match`: post-match score filter (default `0.01` like wrapper)

## Common pitfalls

- Using `raw_detect * mask` before top-k (breaks keypoint selection when scores are negative).
- Forgetting border suppression (usually increases unstable keypoints near edges).
- Comparing descriptor tensors without aligning keypoints first (descriptor MSE becomes meaningless if ordering changes).
- FP16 simplification warnings from `onnxsim`: always validate FP16 with `check_onnx.py`.
