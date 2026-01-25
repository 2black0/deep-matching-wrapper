from __future__ import annotations

from typing import Iterable


def ort_providers(device: str | None) -> list[str]:
    # Best-effort CUDA; ORT will silently fall back to CPU when unavailable.
    if device is None:
        device = "cuda"
    device = str(device).lower()
    if device.startswith("cuda") or device.startswith("gpu"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
