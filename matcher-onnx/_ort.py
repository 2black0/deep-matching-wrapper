from __future__ import annotations

def ort_providers(device: str | None) -> list[str]:
    # Best-effort CUDA; ORT will silently fall back to CPU when unavailable.
    if device is None:
        device = "cuda"
    device = str(device).lower()
    if device.startswith("cuda") or device.startswith("gpu"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_session(model_path: str, device: str | None):
    """Create an ORT session with sane defaults.

    Note: ORT may add Memcpy nodes when mixing CPU/GPU tensors. Those are logged
    at warning level; we silence warnings by default to keep CLI output clean.
    """

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 0=VERBOSE 1=INFO 2=WARNING 3=ERROR 4=FATAL
    so.log_severity_level = 3

    return ort.InferenceSession(str(model_path), sess_options=so, providers=ort_providers(device))
