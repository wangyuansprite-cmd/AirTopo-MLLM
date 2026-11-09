"""Formal AirTopo-MLLM voxelwise and structural evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


class MetricError(RuntimeError):
    """Raised when aligned evaluation inputs violate the metric contract."""


def _aligned(*arrays: np.ndarray) -> None:
    shapes = {tuple(array.shape) for array in arrays}
    if len(shapes) != 1:
        raise MetricError(f"array shape mismatch: {sorted(shapes)}")


def compute_dlr_arrays(
    prediction: np.ndarray,
    centerline: np.ndarray,
    branch_map: np.ndarray,
) -> dict[str, Any]:
    """Compute centerline-voxel-length Detected Length Ratio."""
    _aligned(prediction, centerline, branch_map)
    pred = prediction.astype(bool, copy=False)
    cl = centerline.astype(bool, copy=False)
    if not np.issubdtype(branch_map.dtype, np.integer) or np.any(branch_map < 0):
        raise MetricError("branch ownership map must contain nonnegative integers")
    if not np.any(cl):
        raise MetricError("GT centerline is empty")
    unowned = int(np.logical_and(cl, branch_map == 0).sum())
    if unowned:
        raise MetricError(f"{unowned} GT centerline voxels have background ownership")
    positive = branch_map > 0
    all_values = np.unique(branch_map[positive]).astype(np.int64, copy=False)
    detected_values = np.unique(branch_map[np.logical_and(pred, positive)])
    detected_values = detected_values[detected_values > 0].astype(np.int64, copy=False)
    centerline_values = branch_map[cl].astype(np.int64, copy=False)
    lengths = np.bincount(centerline_values, minlength=int(all_values.max()) + 1)
    total_length = int(lengths[all_values].sum())
    detected_length = int(lengths[detected_values].sum())
    return {
        "total_branches": int(all_values.size),
        "detected_branches": int(detected_values.size),
        "total_length": total_length,
        "detected_length": detected_length,
        "dlr": float(detected_length / total_length),
        "length_unit": "GT_centerline_voxel_count",
    }


def compute_airway_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    centerline: np.ndarray,
    branch_map: np.ndarray,
) -> dict[str, Any]:
    """Compute Dice, IoU, precision, recall, CLR, DBR, and DLR."""
    _aligned(prediction, target, centerline, branch_map)
    pred = prediction.astype(bool, copy=False)
    truth = target.astype(bool, copy=False)
    cl = centerline.astype(bool, copy=False)
    if not np.any(truth) or not np.any(cl):
        raise MetricError("target and centerline must be nonempty")
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    gt_values = np.unique(branch_map[truth])
    gt_values = gt_values[gt_values > 0]
    detected_values = np.unique(branch_map[np.logical_and(pred, truth)])
    detected_values = detected_values[detected_values > 0]
    if gt_values.size == 0:
        raise MetricError("target contains no positive ownership IDs")
    dlr = compute_dlr_arrays(pred, cl, branch_map)
    return {
        "dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "iou": float(tp / max(tp + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "clr": float(np.logical_and(pred, cl).sum() / int(cl.sum())),
        "dbr": float(detected_values.size / gt_values.size),
        "dlr": dlr["dlr"],
        "prediction_voxels": int(pred.sum()),
        "target_voxels": int(truth.sum()),
        "gt_branches": int(gt_values.size),
        "detected_branches": int(detected_values.size),
        "total_branch_centerline_voxels": dlr["total_length"],
        "detected_branch_centerline_voxels": dlr["detected_length"],
    }

