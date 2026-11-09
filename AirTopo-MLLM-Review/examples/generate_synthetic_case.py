#!/usr/bin/env python3
"""Generate a non-clinical airway example in the released data format.

All arrays are created procedurally. No patient image, clinical metadata, or
source-dataset voxel is read by this script.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


OUTPUT_DIR = Path(__file__).resolve().parent / "synthetic_case"
SHAPE_DHW = (64, 96, 96)
SPACING_XYZ = (0.8, 0.8, 0.8)
ORIGIN_LPS = (-38.4, -38.4, -25.6)
DIRECTION_LPS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
SEED = 2026


BRANCHES = (
    {"branch_id": 1, "generation": 0, "side": "trachea", "start": (4, 48, 48), "end": (27, 48, 48)},
    {"branch_id": 2, "generation": 1, "side": "left", "start": (27, 48, 48), "end": (43, 59, 68)},
    {"branch_id": 3, "generation": 1, "side": "right", "start": (27, 48, 48), "end": (43, 59, 28)},
    {"branch_id": 4, "generation": 2, "side": "left", "start": (43, 59, 68), "end": (57, 72, 80)},
    {"branch_id": 5, "generation": 2, "side": "left", "start": (43, 59, 68), "end": (57, 49, 80)},
    {"branch_id": 6, "generation": 2, "side": "right", "start": (43, 59, 28), "end": (57, 72, 16)},
    {"branch_id": 7, "generation": 2, "side": "right", "start": (43, 59, 28), "end": (57, 49, 16)},
)


def segment_points(start: tuple[int, int, int], end: tuple[int, int, int]) -> np.ndarray:
    """Return rounded, unique DHW samples along one straight segment."""
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    count = int(np.max(np.abs(last - first))) + 1
    points = np.rint(np.linspace(first, last, count)).astype(np.int64)
    return np.unique(points, axis=0)


def make_image(array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    image.SetSpacing(SPACING_XYZ)
    image.SetOrigin(ORIGIN_LPS)
    image.SetDirection(DIRECTION_LPS)
    image.SetMetaData("descrip", "Procedural synthetic airway example; no clinical source")
    return image


def write_image(path: Path, array: np.ndarray) -> None:
    sitk.WriteImage(make_image(array), str(path), True)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    centerline = np.zeros(SHAPE_DHW, dtype=np.uint8)
    centerline_owner = np.zeros(SHAPE_DHW, dtype=np.int16)
    branch_records = []
    for record in BRANCHES:
        points = segment_points(record["start"], record["end"])
        indices = tuple(points[:, axis] for axis in range(3))
        centerline[indices] = 1
        # At a shared junction, preserve the more proximal branch ownership.
        empty = centerline_owner[indices] == 0
        centerline_owner[tuple(axis[empty] for axis in indices)] = record["branch_id"]
        branch_records.append({
            **record,
            "start": list(record["start"]),
            "end": list(record["end"]),
            "centerline_voxels_before_junction_deduplication": int(len(points)),
        })

    distance, nearest = distance_transform_edt(
        centerline == 0,
        sampling=(SPACING_XYZ[2], SPACING_XYZ[1], SPACING_XYZ[0]),
        return_indices=True,
    )
    airway = distance <= 2.8
    branch_map = centerline_owner[tuple(nearest)].astype(np.int16, copy=False)
    branch_map[~airway] = 0
    if np.any(np.logical_and(centerline > 0, branch_map == 0)):
        raise RuntimeError("synthetic centerline contains unowned voxels")

    rng = np.random.default_rng(SEED)
    ct = rng.normal(-0.15, 0.025, size=SHAPE_DHW).astype(np.float32)
    ct[airway] = rng.normal(-0.85, 0.02, size=int(airway.sum())).astype(np.float32)
    ct = np.clip(ct, -1.0, 1.0)

    write_image(OUTPUT_DIR / "ct.nii.gz", ct)
    write_image(OUTPUT_DIR / "whole_airway_mask.nii.gz", airway.astype(np.uint8))
    write_image(OUTPUT_DIR / "semantic_centerline.nii.gz", centerline)
    write_image(OUTPUT_DIR / "branch_id_map.nii.gz", branch_map)

    manifest_fields = [
        "case_id", "target_id", "ct_path", "mask_path", "centerline_path",
        "branch_map_path", "prompt",
    ]
    manifest_row = {
        "case_id": "SYNTHETIC_001",
        "target_id": "whole_airway",
        "ct_path": "ct.nii.gz",
        "mask_path": "whole_airway_mask.nii.gz",
        "centerline_path": "semantic_centerline.nii.gz",
        "branch_map_path": "branch_id_map.nii.gz",
        "prompt": "segment the entire airway tree",
    }
    with (OUTPUT_DIR / "sample_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerow(manifest_row)

    metadata = {
        "case_id": "SYNTHETIC_001",
        "synthetic": True,
        "clinical_source": None,
        "patient_identifier": None,
        "generator": "examples/generate_synthetic_case.py",
        "seed": SEED,
        "shape_dhw": list(SHAPE_DHW),
        "spacing_xyz_mm": list(SPACING_XYZ),
        "origin_lps_mm": list(ORIGIN_LPS),
        "direction_lps": list(DIRECTION_LPS),
        "ct_dtype": str(ct.dtype),
        "mask_dtype": "uint8",
        "branch_map_dtype": str(branch_map.dtype),
        "airway_voxels": int(airway.sum()),
        "centerline_voxels": int(centerline.sum()),
        "branch_count": len(BRANCHES),
        "note": "Procedural format example only; not a clinical or benchmark case.",
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "branch_metadata.json").write_text(
        json.dumps({"branches": branch_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote synthetic example to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
