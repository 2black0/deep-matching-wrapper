# XFeat ONNX Export

This folder contains scripts and notes for exporting XFeat to ONNX and validating the exported models against the PyTorch baseline.

XFeat in this repository supports three user-facing modes:

- `xfeat` (sparse): detector + descriptor, then mutual-nearest-neighbor matching.
- `xfeat-star` (semi-dense): multi-scale dense extraction + refinement (not exported end-to-end here).
- `xfeat-lightglue`: XFeat detector+descriptor followed by a LightGlue-style matcher (“LighterGlue”).

Because ONNX does not handle Python control flow and variable-length list outputs well, the conversion is split into ONNX-friendly components.

Files:
- `matcher/xfeat/onnx/convert-onnx.py`: export ONNX models
- `matcher/xfeat/onnx/check-onnx.py`: compare ONNX vs PyTorch numerically
- `matcher/xfeat/onnx/demo_onnx.py`: run end-to-end matching using ONNX Runtime

ONNX models are written to `matcher/xfeat/weights/`.

## Architecture overview (as implemented here)

Core modules:
- `matcher/xfeat/modules/model.py`: `XFeatModel` CNN backbone + heads
- `matcher/xfeat/modules/xfeat.py`: full pipeline (preprocess, heatmap, NMS, interpolation, matching)
- `matcher/xfeat/modules/lighterglue.py`: loads a smaller LightGlue configuration for XFeat descriptors

### XFeatModel forward

`XFeatModel.forward(image)` (see `matcher/xfeat/modules/model.py`) returns:

- `feats`: dense descriptor map `(B, 64, H/8, W/8)`
- `keypoints`: logits `(B, 65, H/8, W/8)`
- `heatmap`: reliability map `(B, 1, H/8, W/8)`

The `keypoints` tensor represents an 8x8 cell classification (64 bins) + a “dustbin” channel (65th). The sparse keypoint heatmap is reconstructed by applying softmax over the 65 channels and reshaping the 64 bins into `(H, W)` at pixel resolution.

### Sparse pipeline in PyTorch

`XFeat.detectAndCompute(...)` (see `matcher/xfeat/modules/xfeat.py`) performs:

1) Resize to multiples of 32
- `preprocess_tensor()` resizes with bilinear interpolation (`align_corners=False`).

2) Backbone inference
- `M1, K1, H1 = net(x)`
- `M1` is L2-normalized channel-wise.

3) Keypoint heatmap reconstruction
- `K1h = get_kpts_heatmap(K1)`
- This is a sensitive reshape/permutation; the order must match exactly.

4) NMS peak extraction
- maxpool NMS on `K1h`.

5) Reliability score
- `score = nearest(K1h at kpts) * bilinear(H1 at kpts)`

6) Top-k selection

7) Descriptor sampling
- bicubic interpolation of `M1` at selected keypoints.

8) Matching
- `xfeat`: mutual nearest neighbor on cosine similarity.
- `xfeat-lightglue`: LightGlue-style matching.

Steps (3–8) include variable-length operations (`nonzero`, padding lists, etc.), so we keep them outside ONNX.

## ONNX export strategy

We export two independent ONNX models:

1) XFeat backbone ONNX
- Exports only the CNN outputs (`descriptors_map`, `kpt_logits`, `reliability`).
- Post-processing (heatmap, NMS, top-k, sampling, matching) is done in Python.

2) LighterGlue ONNX (optional)
- Exports the LightGlue-style matcher core.
- Input: fixed number of keypoints/descriptors (`k`), padded to a fixed size.
- Output: `log_assignment` matrix `(B, k+1, k+1)`.
- Discrete match indices are computed outside ONNX using the same `filter_matches` logic as LightGlue.

This split is deliberate for academic correctness and engineering practicality:
- It avoids exporting dynamic control flow.
- It makes it easy to isolate and validate each component.

## 1) Convert / Export

### Export XFeat backbone

```bash
pixi run python matcher/xfeat/onnx/convert-onnx.py \
  --matcher xfeat \
  --dtype FP32 \
  --size 640 480

pixi run python matcher/xfeat/onnx/convert-onnx.py \
  --matcher xfeat \
  --dtype FP16 \
  --size 640 480
```

Outputs:
- `matcher/xfeat/weights/xfeat_backbone_fp32_640x480.onnx`
- `matcher/xfeat/weights/xfeat_backbone_fp16_640x480.onnx`

### Export LighterGlue matcher

Because LightGlue expects a fixed tensor size for ONNX export, we export a model per `k`:

```bash
pixi run python matcher/xfeat/onnx/convert-onnx.py \
  --matcher lightglue \
  --dtype FP32 \
  --num-kpts 1024 \
  --size 640 480

pixi run python matcher/xfeat/onnx/convert-onnx.py \
  --matcher lightglue \
  --dtype FP16 \
  --num-kpts 1024 \
  --size 640 480
```

Outputs:
- `matcher/xfeat/weights/xfeat_lighterglue_fp32_k1024.onnx`
- `matcher/xfeat/weights/xfeat_lighterglue_fp16_k1024.onnx`

### LightGlue export note (important)

Kornia’s `LightGlue.forward/_forward` contains runtime assertions like:

- `torch.all(kpts >= -1).item()`

These `.item()` calls are not compatible with `torch.export` tracing.

For that reason, the exporter does NOT call `LightGlue.forward`. Instead it re-implements the core computation:

- normalize keypoints
- project descriptors (`input_proj`)
- positional encoding (`posenc`)
- transformer stack
- log assignment computation

This yields a stable tensor output (`log_assignment`) that can be exported.

## 2) Numerical check (PyTorch vs ONNX)

### Check XFeat backbone

```bash
pixi run python matcher/xfeat/onnx/check-onnx.py \
  --matcher xfeat \
  --dtype FP32 \
  --size 640 480
```

Reported metrics (MSE):
- descriptor map
- keypoint logits
- reliability map

### Check LighterGlue

```bash
pixi run python matcher/xfeat/onnx/check-onnx.py \
  --matcher lightglue \
  --dtype FP32 \
  --num-kpts 1024 \
  --size 640 480
```

Reported metric:
- `log_assignment` MSE

Interpretation:
- FP32 should be small (typically ~1e-4 to 1e-9 depending on backend and attention kernels).
- FP16 will generally have higher error; for academic parity claims, FP32 is the recommended reference.

## 3) ONNX demo (end-to-end matching)

The demo runs the exported backbone and then performs the same post-processing as `XFeat.detectAndCompute`:

- reconstruct heatmap from logits
- NMS
- reliability scoring
- top-k
- descriptor sampling

Then it matches with either MNN (`xfeat`) or LighterGlue (`lightglue`).

### Demo: XFeat (MNN)

```bash
pixi run python matcher/xfeat/onnx/demo_onnx.py \
  --matcher xfeat \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --dtype fp32 \
  --size 640 480 \
  --top-k 1024
```

### Demo: XFeat + LightGlue

```bash
pixi run python matcher/xfeat/onnx/demo_onnx.py \
  --matcher lightglue \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --dtype fp32 \
  --size 640 480 \
  --top-k 1024 \
  --min-conf 0.1
```

Output:
- `matcher/xfeat/onnx/result.jpg`

By default the visualization draws only RANSAC inliers. Use `--draw-all` to draw all matches.

## Common pitfalls

- Heatmap reconstruction permute/reshape order mismatch: produces very wrong keypoints.
- Sampling coordinates with `align_corners=True` math: shifts descriptors and breaks matching.
- Forgetting that ONNX LightGlue export is fixed-`k`: the demo must use the same `--top-k` as the exported matcher.
- FP16 mismatch: expect larger numeric drift in attention-heavy modules.
