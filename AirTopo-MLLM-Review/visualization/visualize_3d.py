"""Render aligned ground-truth and prediction masks in a shared 3D view."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion


BACKGROUND_COLOR = "#C9CDD2"
GROUND_TRUTH_COLOR = "#2A9D8F"
PREDICTION_COLOR = "#55A868"


def boundary_coordinates(mask: np.ndarray, max_points: int, seed: int = 2026) -> np.ndarray:
    """Return deterministically sampled boundary coordinates in DHW order."""
    binary = mask.astype(bool, copy=False)
    boundary = np.logical_and(binary, ~binary_erosion(binary))
    coordinates = np.argwhere(boundary)
    if len(coordinates) > max_points:
        rng = np.random.default_rng(seed)
        coordinates = coordinates[rng.choice(len(coordinates), max_points, replace=False)]
    return coordinates


def _draw(ax, coordinates: np.ndarray, color: str, size: float, alpha: float) -> None:
    if coordinates.size:
        ax.scatter(
            coordinates[:, 2], coordinates[:, 1], coordinates[:, 0],
            s=size, c=color, alpha=alpha, linewidths=0, rasterized=True,
        )


def render_airway_comparison(
    whole_airway: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_png: Path,
    output_pdf: Path,
    title: str,
    elev: float = 18.0,
    azim: float = -72.0,
    dpi: int = 600,
    max_context_points: int = 100_000,
    max_target_points: int = 80_000,
) -> None:
    """Save matched ground-truth and prediction panels with shared limits."""
    if not (whole_airway.shape == ground_truth.shape == prediction.shape):
        raise ValueError("whole-airway, ground-truth, and prediction shapes differ")
    context = boundary_coordinates(whole_airway, max_context_points)
    truth = boundary_coordinates(ground_truth, max_target_points)
    pred = boundary_coordinates(prediction, max_target_points)
    occupied = np.argwhere(whole_airway > 0)
    if not len(occupied):
        raise ValueError("whole-airway context is empty")
    low = occupied.min(axis=0)
    high = occupied.max(axis=0)
    figure = plt.figure(figsize=(10.5, 5.2), facecolor="white")
    for index, (label, target, color) in enumerate(
        (("Ground truth", truth, GROUND_TRUTH_COLOR),
         ("Prediction", pred, PREDICTION_COLOR)), 1
    ):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        _draw(axis, context, BACKGROUND_COLOR, 0.20, 0.12)
        _draw(axis, target, color, 0.85, 0.95)
        axis.set_xlim(low[2], high[2]); axis.set_ylim(low[1], high[1]); axis.set_zlim(low[0], high[0])
        axis.view_init(elev=elev, azim=azim)
        axis.set_box_aspect((high[2]-low[2]+1, high[1]-low[1]+1, high[0]-low[0]+1))
        axis.set_axis_off(); axis.set_title(label, fontsize=13, weight="bold")
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)

