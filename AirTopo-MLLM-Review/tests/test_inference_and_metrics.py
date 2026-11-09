"""Non-sensitive numerical tests for reconstruction and structural metrics."""

import numpy as np

from evaluation.metrics import compute_airway_metrics
from inference.volume_inference import sliding_window_probability


def test_sliding_window_uniform_average_and_metrics():
    volume = np.zeros((6, 7, 8), dtype=np.float32)
    probability, prediction, metadata = sliding_window_probability(
        volume, lambda patch: np.full(patch.shape, 0.75, dtype=np.float32),
        roi_size=(4, 5, 6), stride=(3, 4, 5), threshold=0.5,
    )
    assert np.allclose(probability, 0.75)
    assert prediction.all() and metadata["patch_count"] == 8
    target = np.zeros_like(prediction); target[1:5, 2:6, 2:7] = True
    centerline = np.zeros_like(prediction); centerline[2, 3, 2:7] = True
    branch_map = np.zeros(prediction.shape, dtype=np.int16)
    branch_map[target] = 1
    values = compute_airway_metrics(prediction, target, centerline, branch_map)
    assert values["recall"] == 1.0
    assert values["clr"] == values["dbr"] == values["dlr"] == 1.0

