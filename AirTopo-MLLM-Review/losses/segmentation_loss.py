"""Dice plus adaptive weighted binary cross-entropy segmentation loss."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.nn import functional as F


def dice_weighted_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_loss_function,
    positive_count: Optional[int] = None,
    positive_weight_cap: float = 20.0,
):
    """Return the formal semantic loss and its two components.

    The positive weight is ``min(sqrt(negative / positive), cap)``. Empty
    targets use a denominator of one, matching the reviewed training code.
    """
    truth = target.float()
    positive = int(truth.sum().item()) if positive_count is None else int(positive_count)
    negative = int(truth.numel()) - positive
    positive_weight = min(
        math.sqrt(negative / max(positive, 1)), float(positive_weight_cap)
    )
    # Preserve the checkpoint model's existing Dice implementation instead of
    # introducing a second formula during release refactoring.
    dice = dice_loss_function(logits, truth)
    weighted_bce = F.binary_cross_entropy_with_logits(
        logits.float(), truth,
        pos_weight=torch.tensor(positive_weight, device=logits.device),
    )
    return dice + weighted_bce, dice, weighted_bce, positive_weight

