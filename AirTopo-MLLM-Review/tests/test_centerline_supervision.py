"""CPU-only tests for semantic-centerline recall supervision."""

import torch

from losses.centerline_supervision import centerline_recall_loss


def test_centerline_recall_loss_and_shape_guard():
    centerline = torch.ones((2, 1, 2, 2, 2), dtype=torch.float32)
    high = torch.full_like(centerline, 8.0, requires_grad=True)
    low = torch.full_like(centerline, -8.0, requires_grad=True)
    high_loss, high_stats = centerline_recall_loss(high, centerline)
    low_loss, _ = centerline_recall_loss(low, centerline)
    assert high_loss.item() < 0.001 < low_loss.item()
    assert high_stats["voxel_count"] == 16
    empty_logits = torch.randn((2, 1, 2, 2, 2), requires_grad=True)
    empty_loss, empty_stats = centerline_recall_loss(empty_logits, torch.zeros_like(empty_logits))
    assert empty_stats["voxel_count"] == 0 and empty_loss.item() == 0.0
    empty_loss.backward()
    assert torch.count_nonzero(empty_logits.grad) == 0
    try:
        centerline_recall_loss(torch.zeros((1, 1, 2, 2, 2)), torch.zeros((1, 1, 2, 2, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch was not rejected")

