# XFeat-LightGlue C++ Implementation Status

## Current State

**XFeat-LightGlue mode in C++ currently uses a fallback implementation:**
- ✅ **Feature Extraction**: Uses XFeat sparse features (same as Python)
- ⚠️ **Matching**: Falls back to Mutual Nearest Neighbors (MNN) instead of LightGlue transformer

## Why?

LightGlue is a complex transformer-based matcher from Kornia that:
1. Has dynamic control flow (early stopping, point pruning)
2. Uses FlashAttention optimizations
3. Is difficult to export to TorchScript without losing functionality

## Results Comparison

| Metric | Python (LightGlue) | C++ (MNN Fallback) | Difference |
|--------|-------------------|-------------------|------------|
| Matches | 822 | 832 | +1.2% |
| Inliers | 717 | 157 | **-78%** ⚠️ |
| Ratio | 87% | 19% | **-68%** ⚠️ |
| Speed (CPU) | N/A | 359ms | - |

**Key Issue**: The MNN fallback produces significantly fewer inliers because it uses simple cosine similarity matching instead of the sophisticated attention-based matching that LightGlue provides.

## Future Improvements

### Option 1: ONNX Runtime (Recommended)
- Export LightGlue to ONNX format
- Use ONNX Runtime for inference
- **Pros**: Good performance, cross-platform
- **Cons**: Requires ONNX export work

### Option 2: Pure C++ Implementation
- Reimplement transformer layers in C++
- Use optimized libraries (cuBLAS, cuDNN)
- **Pros**: Maximum control, no dependencies
- **Cons**: **Very complex** (weeks of work)

### Option 3: Simplified LightGlue
- Create a distilled version without dynamic features
- Make it TorchScript-compatible
- **Pros**: Clean integration
- **Cons**: May lose some accuracy

## Current Usage

The C++ code automatically detects when LightGlue TorchScript model is missing and falls back to MNN:

```cpp
// From XFeatTorchMatcher.cpp:131-152
} else if (cfg.mode == XFeatMode::XFEAT_LIGHTGLUE) {
  const std::string lg_path = base + "/xfeat_lightglue_fp32_k" + std::to_string(cfg.top_k) + ".pt";
  if (std::filesystem::exists(lg_path)) {
    // Load LightGlue TorchScript (not yet implemented)
    module_xfeat_lightglue = torch::jit::load(lg_path, device);
    has_lightglue_model = true;
  } else {
    // Fallback: use sparse XFeat + MNN
    module_xfeat = torch::jit::load(base + "/xfeat_fp32_k" + std::to_string(cfg.top_k) + ".pt", device);
  }
}
```

## Workaround

For production use requiring high accuracy:
1. **Use Python for matching-critical applications**
2. **Use XFeat-Star for C++ deployments** (53% inlier ratio, much better than MNN)
3. **Wait for ONNX Runtime integration** (future enhancement)

## Related Files

- `matcher/xfeat/modules/lighterglue.py` - Python LightGlue wrapper
- `matcher/xfeat/weights/xfeat-lighterglue.pt` - Transformer weights
- `matcher-cpp/xfeat/src/XFeatTorchMatcher.cpp:357-430` - Fallback implementation

---

**Last Updated**: January 27, 2026
**Status**: Fallback implementation only (MNN matching)
