# CLIDD ONNX Conversion & Verification

This directory contains scripts to export CLIDD models to ONNX format (FP32/FP16) and verify their numerical accuracy against the PyTorch baseline.

## 📂 Project Structure

- **`onnx/convert_onnx.py`**: Exports PyTorch `.pth` weights to optimized `.onnx` files.
- **`onnx/check_onnx.py`**: Compares output tensors (Keypoints, Scores, Descriptors) between PyTorch and ONNX.
- **`weights/`**: Stores source `.pth` files and generated `.onnx` / `.onnx.data` files.

---

## 🚀 How to Use

### 1. Convert to ONNX

Run this script to generate optimized ONNX models. You can choose between **FP32** (for Hailo-8 base/High precision) or **FP16** (for Raspberry Pi CPU speed).

```bash
# Convert to FP32 (Recommended for Hailo-8 NPU input)
pixi run python matcher/clidd/onnx/convert_onnx.py --weights A48 --dtype FP32

# Convert to FP16 (Recommended for RPi CPU inference)
pixi run python matcher/clidd/onnx/convert_onnx.py --weights A48 --dtype FP16

```

**Parameters:**

- `--weights`: Model variant (`A48`, `S64`, `U128`, etc.)
- `--dtype`: `FP32` or `FP16`
- `--size`: Input resolution (Default: `640 480`)
- `--topk`: Number of keypoints (Default: `1024`)

### 2. Verify Accuracy

Compare the generated ONNX models against the original PyTorch implementation to ensure stability.

```bash
pixi run python matcher/clidd/onnx/check_onnx.py --weights A48

```

---

## 📊 Understanding Results

| Metric                     | Target (Good Result) | Description                                                              |
| -------------------------- | -------------------- | ------------------------------------------------------------------------ |
| **Spatial Precision @3px** | **> 95%**            | Percentage of points matching the original location within a 3px radius. |
| **Mean Point Dist (px)**   | **< 1.0 px**         | The average pixel shift of detected keypoints.                           |
| **Scores MSE**             | **< 1e-3**           | Difference in confidence values.                                         |
| **Descriptor MSE**         | **< 2e-2**           | Numerical fidelity of the feature vectors.                               |

> **Note:** A `Max diff` warning during simplification is normal for FP16 due to index shifting in low-score keypoints. As long as **Spatial Precision** remains high, the model is valid.