"""Uniform-overlap full-volume inference used by AirTopo-MLLM.

The model-specific patch callable is injected by the caller. This keeps the
review utility independent of private checkpoints while preserving the formal
ROI traversal, probability averaging, and strict thresholding behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

import numpy as np


def sliding_starts(size: int, roi: int, stride: int) -> list[int]:
    """Return complete sliding-window starts, including an aligned final patch."""
    if roi <= 0 or stride <= 0:
        raise ValueError(f"ROI and stride must be positive, got roi={roi}, stride={stride}")
    if size < roi:
        raise ValueError(f"volume axis size {size} is smaller than ROI axis {roi}")
    starts = list(range(0, size - roi + 1, stride))
    last = size - roi
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def sliding_window_probability(
    volume: np.ndarray,
    predict_patch: Callable[[np.ndarray], np.ndarray],
    roi_size: Sequence[int] = (128, 192, 192),
    stride: Sequence[int] = (64, 96, 96),
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct a full probability map by uniform overlap averaging."""
    if volume.ndim != 3:
        raise ValueError(f"expected a DHW volume, got shape {volume.shape}")
    roi = tuple(int(value) for value in roi_size)
    step = tuple(int(value) for value in stride)
    if len(roi) != 3 or len(step) != 3:
        raise ValueError("ROI and stride must each have three values")
    axes = [sliding_starts(size, width, hop) for size, width, hop in zip(volume.shape, roi, step)]
    probability_sum = np.zeros(volume.shape, dtype=np.float32)
    weight_sum = np.zeros(volume.shape, dtype=np.float32)
    completed = 0
    for d0, h0, w0 in product(*axes):
        slices = (
            slice(d0, d0 + roi[0]),
            slice(h0, h0 + roi[1]),
            slice(w0, w0 + roi[2]),
        )
        patch = np.ascontiguousarray(volume[slices])
        probability = np.asarray(predict_patch(patch), dtype=np.float32)
        if probability.shape != roi:
            raise ValueError(f"patch output {probability.shape} != ROI {roi}")
        if not np.isfinite(probability).all():
            raise ValueError("patch probability contains NaN or Inf")
        probability_sum[slices] += probability
        weight_sum[slices] += 1.0
        completed += 1
    if np.any(weight_sum == 0):
        raise RuntimeError("full-volume coverage is incomplete")
    final_probability = probability_sum / weight_sum
    prediction = final_probability > float(threshold)
    metadata = {
        "roi_size_dhw": list(roi),
        "stride_dhw": list(step),
        "patch_count": completed,
        "fusion": "uniform_probability_average",
        "threshold_rule": f"probability > {float(threshold)}",
        "axis_starts_dhw": axes,
    }
    return final_probability, prediction, metadata

