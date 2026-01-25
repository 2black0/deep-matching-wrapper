# LiftFeat ONNX Export

This folder contains scripts to export the LiftFeat network to ONNX and validate that the exported model behaves the same as the PyTorch implementation.

Files:
- `matcher/liftfeat/onnx/convert_onnx.py`: export LiftFeat to ONNX (FP32/FP16)
- `matcher/liftfeat/onnx/check_onnx.py`: numerical comparison between PyTorch and ONNX
- `matcher/liftfeat/onnx/demo_onnx.py`: end-to-end keypoints + matching demo using ONNX Runtime

The ONNX models are written to `matcher/liftfeat/weights/`.

## Background: what is exported

LiftFeat has two main stages (see `matcher/liftfeat/modules/model.py`):

1) `LiftFeatSPModel.forward1(image)`
- Input image is converted to grayscale and normalized with `InstanceNorm2d(1)`.
- The network produces:
  - `des_map`: dense descriptor map, shape `(B, 64, H/8, W/8)`
  - `kpt_map`: keypoint logits, shape `(B, 65, H/8, W/8)`
  - `d_feats`: depth/normal features, shape `(B, C, H/8, W/8)`

2) `LiftFeatSPModel.forward2(des_map, kpt_map, d_feats)`
- Unfolds `d_feats` into per-cell "normal" features.
- Runs the FeatureBooster and produces refined descriptors (vector form).

The PyTorch wrapper (`matcher/liftfeat/modules/liftfeat_wrapper.py`) then:
- reshapes refined descriptors back into a descriptor map and L2-normalizes it
- converts `kpt_map` logits into a pixel-level heatmap via softmax + reshape
- applies NMS on the heatmap and samples scores/descriptors using bicubic `grid_sample(align_corners=False)`

## Export format (important design choice)

The exported ONNX model outputs:
- `kpt_logits`: the raw keypoint logits (same as `kpt_map`), shape `(1, 65, H/8, W/8)`
- `descriptors_map`: the refined descriptor map, shape `(1, 64, H/8, W/8)`

We intentionally export logits (not the post-softmax heatmap). This avoids small numerical/layout differences across backends affecting detection, and lets downstream code reconstruct the heatmap exactly like the PyTorch wrapper.

## 1) Convert / Export

From the repository root:

```bash
pixi run python matcher/liftfeat/onnx/convert_onnx.py \
  --weights matcher/liftfeat/weights/LiftFeat.pth \
  --size 640 480 \
  --dtype FP32

pixi run python matcher/liftfeat/onnx/convert_onnx.py \
  --weights matcher/liftfeat/weights/LiftFeat.pth \
  --size 640 480 \
  --dtype FP16
```

Outputs:
- `matcher/liftfeat/weights/liftfeat_fp32_640x480.onnx`
- `matcher/liftfeat/weights/liftfeat_fp16_640x480.onnx`

Notes:
- The exporter uses opset 18.
- The script runs `onnxsim` to simplify the graph.
- Depending on your ONNX save settings, you may also see external tensor data files:
  - `matcher/liftfeat/weights/*.onnx.data`
  Keep these next to the `.onnx` file.

### Exporter implementation details (academic notes)

The exporter wraps the original model in `LiftFeatExport` and implements the same tensor transformations as the PyTorch wrapper.

Key points that must match PyTorch exactly:

1) Unfold ordering (critical)
- LiftFeat uses a custom `_unfold2d()` in `matcher/liftfeat/modules/model.py`.
- The channel ordering after unfold is not the same as a naive `.unfold(...).view(...)`.
- In the exporter, `_unfold2d_onnx()` replicates:
  - reshape to `(B, C, H/ws, W/ws, ws**2)`
  - permute to `(B, C, ws**2, H/ws, W/ws)`
  - reshape to `(B, C*ws**2, H/ws, W/ws)`

If unfold ordering differs, the FeatureBooster sees permuted features and the refined descriptors degrade even if the ONNX graph runs successfully.

2) Descriptor map normalization
- The PyTorch wrapper L2-normalizes the descriptor map before sampling.
- The exporter normalizes `descriptors_map` inside the forward path.

3) Heatmap is reconstructed outside the model
- We export `kpt_logits` and reconstruct:
  - `scores_raw = softmax(kpt_logits, dim=1)[:, :64]`
  - reshape/permute into a `(B, 1, H, W)` heatmap with the same layout as `liftfeat_wrapper.py`

## 2) Numerical check (PyTorch vs ONNX)

Run:

```bash
pixi run python matcher/liftfeat/onnx/check_onnx.py
```

What it does:
- Builds a PyTorch baseline that matches `liftfeat_wrapper.py` logic:
  - `forward1` + `forward2`
  - reshape + normalize descriptor map
  - reconstruct heatmap from logits
- Runs ONNX Runtime on the exported model.
- Reconstructs heatmap from ONNX `kpt_logits` using the same numpy logic.
- Reports MSE for:
  - heatmap
  - descriptor map

Interpretation:
- Descriptor MSE should be very small (typically ~1e-9 to ~1e-7 for FP32).
- Heatmap MSE can be noticeably larger because softmax + very sparse probabilities make the metric sensitive.
  In practice, the best validation is end-to-end matching quality (see the demo) rather than heatmap MSE alone.

## 3) ONNX demo (end-to-end matching)

Run:

```bash
pixi run python matcher/liftfeat/onnx/demo_onnx.py \
  --img1 assets/ref.png \
  --img2 assets/tgt.png \
  --weights matcher/liftfeat/weights/liftfeat_fp32_640x480.onnx \
  --detect-threshold 0.005
```

Outputs:
- `matcher/liftfeat/onnx/result.jpg`

Important behavior notes:
- By default the visualization draws only RANSAC inliers (cleaner for inspection).
- To draw all raw matches (including outliers), add `--draw-all`.

Parameters:
- `--detect-threshold`: NMS threshold on the heatmap. A value around `0.005` is often reasonable for this demo.
- `--top-k`: limits keypoints by sampled score for runtime sanity.
- `--min-cossim`: optional cosine similarity threshold on mutual nearest neighbor matches.
- `--debug`: prints heatmap statistics and peak counts.

Implementation notes (to match PyTorch):
- Heatmap reconstruction uses the exact permute/reshape order from `matcher/liftfeat/modules/liftfeat_wrapper.py`.
- Sampling uses a mapping equivalent to `torch.grid_sample(..., align_corners=False)`.
  This mapping is easy to get wrong; using `align_corners=True` style math will shift samples and degrade matching.

## Common pitfalls

- Wrong unfold ordering for `d_feats` -> FeatureBooster produces poor descriptors.
- Reconstructing heatmap with a different reshape order -> NMS sees wrong peaks.
- Sampling descriptors/scores with an `align_corners=True` coordinate mapping -> descriptors no longer match PyTorch.
- Using a high detection threshold (e.g. `0.1`) -> almost no keypoints in many images.
