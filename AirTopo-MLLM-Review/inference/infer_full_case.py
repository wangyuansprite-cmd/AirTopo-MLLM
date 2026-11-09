#!/usr/bin/env python3
"""Sanitized full-volume inference entry point for academic review.

The original experiment driver imported private model-construction, tokenizer,
manifest, and checkpoint-loading code. Those integration details are replaced
by an explicit external hook in this public version. The released code retains
the verified volume traversal, probability fusion, thresholding, spatial
validation, metric computation, and NIfTI export behavior.

This file contains no model weights, clinical data paths, credentials, or
private repository layout. It cannot load AirTopo-MLLM without an authorized
integration module supplied by the user.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from evaluation.metrics import compute_airway_metrics
from inference.volume_inference import sliding_window_probability


class RedactedIntegrationError(RuntimeError):
    """Raised when the authorized model integration contract is unavailable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sanitized full-volume inference through a user-supplied, "
            "authorized AirTopo-MLLM integration module."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--integration-module",
        required=True,
        help=(
            "Importable private module exposing load_predictor(). The module "
            "is intentionally not distributed with this review repository."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--roi-size", nargs=3, type=int, default=(128, 192, 192),
        metavar=("D", "H", "W"),
    )
    parser.add_argument(
        "--stride", nargs=3, type=int, default=(64, 96, 96),
        metavar=("D", "H", "W"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--centerline", type=Path)
    parser.add_argument("--branch-map", type=Path)
    parser.add_argument(
        "--record-input-paths",
        action="store_true",
        help="Record full user-supplied input paths in infer_metadata.json.",
    )
    return parser.parse_args()


def load_authorized_predictor(
    module_name: str,
    config_path: Path,
    checkpoint_path: Path,
    prompt: str,
    target_id: str,
    device: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """Load a patch predictor through the redacted model-integration boundary.

    The external module must implement::

        load_predictor(
            config_path: Path,
            checkpoint_path: Path,
            prompt: str,
            target_id: str,
            device: str,
        ) -> Callable[[np.ndarray], np.ndarray]

    The returned callable receives one normalized CT patch in DHW order and
    must return a probability array with the same shape. CT normalization,
    tokenizer construction, model assembly, and checkpoint interpretation
    remain the responsibility of the authorized integration module.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RedactedIntegrationError(
            f"authorized integration module could not be imported: {module_name!r}"
        ) from exc
    factory = getattr(module, "load_predictor", None)
    if not callable(factory):
        raise RedactedIntegrationError(
            f"{module_name!r} must expose a callable load_predictor"
        )
    predictor = factory(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        target_id=target_id,
        device=device,
    )
    if not callable(predictor):
        raise RedactedIntegrationError("load_predictor must return a callable")
    return predictor


def same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    """Return whether two images share size and LPS physical geometry."""
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), atol=1e-6)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), atol=1e-5)
        and np.allclose(first.GetDirection(), second.GetDirection(), atol=1e-6)
    )


def read_aligned_array(
    path: Path,
    reference: sitk.Image,
    *,
    binary: bool,
) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    if not same_geometry(reference, image):
        raise ValueError(f"image geometry does not match CT: {path}")
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D image at {path}, got {array.shape}")
    return array > 0 if binary else array


def write_like(array: np.ndarray, reference: sitk.Image, path: Path) -> None:
    image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path), True)


def input_reference(path: Path, include_full_path: bool) -> str:
    return str(path.resolve()) if include_full_path else path.name


def validate_metric_inputs(args: argparse.Namespace) -> bool:
    supplied = [args.gt is not None, args.centerline is not None, args.branch_map is not None]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "--gt, --centerline, and --branch-map must be supplied together"
        )
    return all(supplied)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.ct.is_file():
        raise FileNotFoundError(args.ct)
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    has_metrics = validate_metric_inputs(args)

    ct_image = sitk.ReadImage(str(args.ct))
    ct = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)
    if ct.ndim != 3:
        raise ValueError(f"expected a 3D CT, got {ct.shape}")

    predictor = load_authorized_predictor(
        args.integration_module,
        args.config,
        args.checkpoint,
        args.prompt,
        args.target_id,
        args.device,
    )
    started = time.time()
    probability, prediction, reconstruction = sliding_window_probability(
        ct,
        predictor,
        roi_size=tuple(args.roi_size),
        stride=tuple(args.stride),
        threshold=float(args.threshold),
    )
    elapsed = time.time() - started
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError("predictor output is not a probability in [0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ct": args.output_dir / "ct_full.nii.gz",
        "probability": args.output_dir / "pred_prob_full.nii.gz",
        "prediction": args.output_dir / "pred_mask_full.nii.gz",
    }
    sitk.WriteImage(ct_image, str(outputs["ct"]), True)
    write_like(probability.astype(np.float32), ct_image, outputs["probability"])
    write_like(prediction.astype(np.uint8), ct_image, outputs["prediction"])

    metric_values = None
    if has_metrics:
        target = read_aligned_array(args.gt, ct_image, binary=True)
        centerline = read_aligned_array(args.centerline, ct_image, binary=True)
        branch_map = read_aligned_array(args.branch_map, ct_image, binary=False)
        if not np.issubdtype(branch_map.dtype, np.integer):
            raise ValueError("branch ownership map must use an integer dtype")
        metric_values = compute_airway_metrics(
            prediction, target, centerline, branch_map
        )
        outputs["ground_truth"] = args.output_dir / "gt_mask_full.nii.gz"
        outputs["metrics"] = args.output_dir / "metrics.json"
        write_like(target.astype(np.uint8), ct_image, outputs["ground_truth"])
        outputs["metrics"].write_text(
            json.dumps(metric_values, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "case_id": args.case_id,
        "target_id": args.target_id,
        "prompt": args.prompt,
        "ct_reference": input_reference(args.ct, args.record_input_paths),
        "config_reference": input_reference(args.config, args.record_input_paths),
        "checkpoint_reference": input_reference(
            args.checkpoint, args.record_input_paths
        ),
        "integration_module": args.integration_module,
        "volume_shape_dhw": list(ct.shape),
        "spacing_xyz": list(ct_image.GetSpacing()),
        "origin_lps": list(ct_image.GetOrigin()),
        "direction_lps": list(ct_image.GetDirection()),
        "elapsed_seconds": elapsed,
        **reconstruction,
        "outputs": {key: value.name for key, value in outputs.items()},
        "metrics_computed": metric_values is not None,
        "release_note": (
            "Model construction and checkpoint interpretation were supplied "
            "by an external authorized integration module."
        ),
    }
    metadata_path = args.output_dir / "infer_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    metadata = run(args)
    print(f"case_id={metadata['case_id']}")
    print(f"target_id={metadata['target_id']}")
    print(f"patch_count={metadata['patch_count']}")
    print(f"output_dir={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
