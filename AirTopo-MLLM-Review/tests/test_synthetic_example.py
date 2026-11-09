from pathlib import Path

import numpy as np
import SimpleITK as sitk

from evaluation.metrics import compute_airway_metrics


ROOT = Path(__file__).parents[1] / "examples" / "synthetic_case"


def test_synthetic_example_geometry_ownership_and_metrics():
    names = (
        "ct.nii.gz",
        "whole_airway_mask.nii.gz",
        "semantic_centerline.nii.gz",
        "branch_id_map.nii.gz",
    )
    images = {name: sitk.ReadImage(str(ROOT / name)) for name in names}
    reference = images["ct.nii.gz"]
    for image in images.values():
        assert image.GetSize() == reference.GetSize()
        assert image.GetSpacing() == reference.GetSpacing()
        assert image.GetOrigin() == reference.GetOrigin()
        assert image.GetDirection() == reference.GetDirection()

    target = sitk.GetArrayFromImage(images["whole_airway_mask.nii.gz"]) > 0
    centerline = sitk.GetArrayFromImage(images["semantic_centerline.nii.gz"]) > 0
    branch_map = sitk.GetArrayFromImage(images["branch_id_map.nii.gz"])
    assert np.any(target)
    assert np.any(centerline)
    assert np.all(branch_map[centerline] > 0)
    assert np.all(branch_map[~target] == 0)

    metrics = compute_airway_metrics(target, target, centerline, branch_map)
    for key in ("dice", "iou", "clr", "dbr", "dlr"):
        assert metrics[key] == 1.0
