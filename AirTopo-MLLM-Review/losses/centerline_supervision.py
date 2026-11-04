"""Semantic-centerline recall supervision on final segmentation logits.

This is the positive-only loss used by the final training stage. It adds no
prediction head and does not resample the semantic centerline target.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def centerline_recall_loss(
    logits: torch.Tensor, semantic_centerline: torch.Tensor
):
    """Return positive-only centerline recall loss and detached diagnostics."""
    centerline = semantic_centerline.to(device=logits.device, dtype=torch.float32)
    if centerline.ndim == 3:
        centerline = centerline.unsqueeze(0).unsqueeze(0)
    elif centerline.ndim == 4:
        centerline = centerline.unsqueeze(1)
    if tuple(centerline.shape) != tuple(logits.shape):
        raise ValueError(
            f"semantic centerline shape {tuple(centerline.shape)} "
            f"!= logits {tuple(logits.shape)}"
        )
    mask = centerline > 0.5
    count = int(mask.sum().item())
    if count == 0:
        zero = logits.float().sum() * 0.0
        return zero, {
            "voxel_count": 0,
            "probability_sum": 0.0,
            "probability_median": 0.0,
            "recall_count_05": 0,
        }
    selected_logits = logits.float()[mask]
    loss = F.softplus(-selected_logits).mean()
    probability = torch.sigmoid(selected_logits.detach())
    return loss, {
        "voxel_count": count,
        "probability_sum": float(probability.sum().item()),
        "probability_median": float(probability.median().item()),
        "recall_count_05": int((probability > 0.5).sum().item()),
    }

