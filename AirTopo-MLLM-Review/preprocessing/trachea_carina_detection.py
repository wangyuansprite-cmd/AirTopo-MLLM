#!/usr/bin/env python3
"""Stage-3 trachea root and carina candidate detection for five test cases.

The detector is read-only with respect to topology_clean case graphs and
topology_raw.  It combines LPS anatomy, label-derived airway caliber, ordered
branch geometry, and rooted graph topology.  It does not assign left/right or
generation labels and does not alter residual cycles.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_PROCESSED_ROOT = DATA_ROOT / "ATM22_processed"
DEFAULT_CLEAN_ROOT = DEFAULT_PROCESSED_ROOT / "topology_clean"
DEFAULT_LABEL_ROOT = DEFAULT_PROCESSED_ROOT / "labelsTr"
DEFAULT_CASES = ()
EDGE_WEIGHT_FLOOR_MM = 0.4
CROP_MARGIN_VOXELS = 16

FIELDS = [
    "case_id",
    "root_node_id",
    "root_lps_x_mm",
    "root_lps_y_mm",
    "root_lps_z_mm",
    "root_incident_branch_id",
    "root_score",
    "root_confidence",
    "root_confidence_label",
    "carina_node_id",
    "carina_lps_x_mm",
    "carina_lps_y_mm",
    "carina_lps_z_mm",
    "carina_score",
    "carina_confidence",
    "carina_confidence_label",
    "trachea_node_ids",
    "trachea_branch_ids",
    "trachea_length_mm",
    "trachea_median_radius_mm",
    "trachea_verticality",
    "trachea_straightness",
    "carina_endpoint_coverage",
    "carina_subtree_balance",
    "carina_child_divergence_angle_deg",
    "carina_lateral_divergence",
    "carina_main_child_branch_ids",
    "residual_cycle_rank_unchanged",
    "needs_manual_review",
    "review_reasons",
    "qc_png",
]


class DetectionError(RuntimeError):
    """Raised when graph, label, or candidate validation fails."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DetectionError(f"{path}: expected JSON object")
    return value


def percentile_rank(values: list[float], value: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float((np.count_nonzero(array < value) + 0.5 * np.count_nonzero(array == value)) / len(array))


def confidence_label(value: float) -> str:
    if value >= 0.80:
        return "high"
    if value >= 0.65:
        return "medium"
    return "low"


def graph_components(
    node_by_id: dict[int, dict[str, Any]],
    branch_by_id: dict[int, dict[str, Any]],
) -> int:
    adjacency = {node_id: set() for node_id in node_by_id}
    for branch in branch_by_id.values():
        start = int(branch["start_node_id"])
        end = int(branch["end_node_id"])
        if start == end:
            raise DetectionError(f"self-loop remains: branch {branch['id']}")
        if start not in adjacency or end not in adjacency:
            raise DetectionError(f"branch {branch['id']} references missing node")
        adjacency[start].add(end)
        adjacency[end].add(start)
    unseen = set(adjacency)
    count = 0
    while unseen:
        count += 1
        seed = min(unseen)
        stack = [seed]
        seen: set[int] = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(adjacency[node_id] - seen)
        unseen -= seen
    return count


def load_graph(case_dir: Path, case_id: str) -> dict[str, Any]:
    nodes_doc = load_json(case_dir / "nodes.json")
    branches_doc = load_json(case_dir / "branches.json")
    stats = load_json(case_dir / "stats.json")
    if nodes_doc.get("case_id") != case_id or branches_doc.get("case_id") != case_id:
        raise DetectionError(f"{case_id}: case ID mismatch")
    if nodes_doc.get("coordinate_system") != "LPS" or branches_doc.get("coordinate_system") != "LPS":
        raise DetectionError(f"{case_id}: graph coordinates are not LPS")
    nodes = nodes_doc["nodes"]
    branches = branches_doc["branches"]
    node_by_id = {int(node["id"]): node for node in nodes}
    branch_by_id = {int(branch["id"]): branch for branch in branches}
    if len(node_by_id) != len(nodes) or len(branch_by_id) != len(branches):
        raise DetectionError(f"{case_id}: duplicate node or branch IDs")
    components = graph_components(node_by_id, branch_by_id)
    rank = len(branch_by_id) - len(node_by_id) + components
    if components != 1 or not stats["clean_is_connected"]:
        raise DetectionError(f"{case_id}: clean graph is disconnected")
    if rank != int(stats["residual_cycle_after"]):
        raise DetectionError(f"{case_id}: residual cycle rank mismatch")

    adjacency: dict[int, list[tuple[int, int, float]]] = {
        node_id: [] for node_id in node_by_id
    }
    for branch_id, branch in branch_by_id.items():
        start = int(branch["start_node_id"])
        end = int(branch["end_node_id"])
        length = float(branch["length_mm"])
        adjacency[start].append((end, branch_id, length))
        adjacency[end].append((start, branch_id, length))
    for edges in adjacency.values():
        edges.sort(key=lambda item: (item[0], item[1]))
    return {
        "nodes": nodes,
        "branches": branches,
        "node_by_id": node_by_id,
        "branch_by_id": branch_by_id,
        "adjacency": adjacency,
        "stats": stats,
        "cycle_rank": rank,
    }


def load_paths_and_radius(
    case_id: str,
    case_dir: Path,
    label_path: Path,
    nodes: list[dict[str, Any]],
    branches: list[dict[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]]]:
    voxel_paths: dict[int, np.ndarray] = {}
    physical_paths: dict[int, np.ndarray] = {}
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    for branch in branches:
        branch_id = int(branch["id"])
        voxels = np.load(
            case_dir / branch["voxel_file"], allow_pickle=False
        ).astype(np.int64, copy=False)
        physical = np.load(
            case_dir / branch["physical_file"], allow_pickle=False
        ).astype(np.float64, copy=False)
        if voxels.shape != (int(branch["n_voxels"]), 3) or physical.shape != voxels.shape:
            raise DetectionError(f"{case_id}: branch {branch_id} array mismatch")
        voxel_paths[branch_id] = voxels
        physical_paths[branch_id] = physical
        lower.append(voxels.min(axis=0))
        upper.append(voxels.max(axis=0))
    for node in nodes:
        voxels = np.asarray(node["voxels"], dtype=np.int64)
        lower.append(voxels.min(axis=0))
        upper.append(voxels.max(axis=0))

    image = sitk.ReadImage(str(label_path))
    if tuple(round(value, 6) for value in image.GetSpacing()) != tuple(
        round(value, 6) for value in (0.8, 0.8, 0.8)
    ):
        raise DetectionError(f"{case_id}: unexpected label spacing {image.GetSpacing()}")
    size = np.asarray(image.GetSize(), dtype=np.int64)
    lo = np.maximum(np.min(np.asarray(lower), axis=0) - CROP_MARGIN_VOXELS, 0)
    hi = np.minimum(np.max(np.asarray(upper), axis=0) + CROP_MARGIN_VOXELS + 1, size)
    full = sitk.GetArrayFromImage(image)
    crop = (
        full[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]] > 0
    )
    del full
    distance = distance_transform_edt(
        crop,
        sampling=tuple(reversed(image.GetSpacing())),
    )

    radius_by_branch: dict[int, dict[str, float]] = {}
    for branch in branches:
        branch_id = int(branch["id"])
        local = voxel_paths[branch_id] - lo
        if np.any(local < 0) or np.any(local >= (hi - lo)):
            raise DetectionError(f"{case_id}: branch {branch_id} outside radius crop")
        radii = distance[local[:, 2], local[:, 1], local[:, 0]]
        radius_by_branch[branch_id] = {
            "median": float(np.median(radii)),
            "p75": float(np.percentile(radii, 75)),
            "max": float(np.max(radii)),
        }
    return physical_paths, radius_by_branch


def endpoint_root_candidates(
    graph: dict[str, Any],
    radius_by_branch: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    node_by_id = graph["node_by_id"]
    adjacency = graph["adjacency"]
    endpoint_ids = sorted(
        node_id for node_id, edges in adjacency.items() if len(edges) == 1
    )
    if len(endpoint_ids) < 2:
        raise DetectionError("graph has fewer than two endpoints")
    endpoint_z = [
        float(node_by_id[node_id]["centroid_physical"][2])
        for node_id in endpoint_ids
    ]
    endpoint_lengths = [
        float(adjacency[node_id][0][2]) for node_id in endpoint_ids
    ]
    endpoint_radii = [
        radius_by_branch[adjacency[node_id][0][1]]["p75"]
        for node_id in endpoint_ids
    ]
    candidates: list[dict[str, Any]] = []
    for node_id in endpoint_ids:
        neighbor, branch_id, length = adjacency[node_id][0]
        root_point = np.asarray(
            node_by_id[node_id]["centroid_physical"], dtype=np.float64
        )
        neighbor_point = np.asarray(
            node_by_id[neighbor]["centroid_physical"], dtype=np.float64
        )
        vector = neighbor_point - root_point
        norm = float(np.linalg.norm(vector))
        inward_inferior_alignment = (
            float(np.clip(-vector[2] / norm, 0.0, 1.0)) if norm > 1e-9 else 0.0
        )
        z_score = percentile_rank(endpoint_z, float(root_point[2]))
        length_score = percentile_rank(endpoint_lengths, float(length))
        radius = radius_by_branch[branch_id]["p75"]
        radius_score = percentile_rank(endpoint_radii, radius)
        score = (
            0.34 * radius_score
            + 0.26 * length_score
            + 0.25 * z_score
            + 0.15 * inward_inferior_alignment
        )
        candidates.append(
            {
                "node_id": node_id,
                "branch_id": branch_id,
                "neighbor_node_id": neighbor,
                "score": float(score),
                "z_score": z_score,
                "length_score": length_score,
                "radius_score": radius_score,
                "inward_inferior_alignment": inward_inferior_alignment,
                "incident_length_mm": float(length),
                "incident_radius_p75_mm": radius,
                "physical_lps": root_point.tolist(),
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["node_id"]))
    return candidates


def rooted_shortest_path_tree(
    root_node_id: int,
    graph: dict[str, Any],
) -> dict[str, Any]:
    adjacency = graph["adjacency"]
    distance = {node_id: float("inf") for node_id in adjacency}
    distance[root_node_id] = 0.0
    parent: dict[int, int | None] = {root_node_id: None}
    parent_branch: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, root_node_id)]
    while queue:
        current_distance, node_id = heapq.heappop(queue)
        if current_distance > distance[node_id] + 1e-12:
            continue
        for neighbor, branch_id, length in adjacency[node_id]:
            weight = max(float(length), EDGE_WEIGHT_FLOOR_MM)
            proposed = current_distance + weight
            current_parent_key = (
                parent.get(neighbor, sys.maxsize),
                parent_branch.get(neighbor, sys.maxsize),
            )
            proposed_parent_key = (node_id, branch_id)
            if (
                proposed < distance[neighbor] - 1e-12
                or (
                    abs(proposed - distance[neighbor]) <= 1e-12
                    and proposed_parent_key < current_parent_key
                )
            ):
                distance[neighbor] = proposed
                parent[neighbor] = node_id
                parent_branch[neighbor] = branch_id
                heapq.heappush(queue, (proposed, neighbor))
    if len(parent) != len(adjacency):
        raise DetectionError("shortest-path tree did not reach all nodes")
    children: dict[int, list[int]] = {node_id: [] for node_id in adjacency}
    for node_id, parent_id in parent.items():
        if parent_id is not None:
            children[parent_id].append(node_id)
    for values in children.values():
        values.sort()
    return {
        "distance": distance,
        "parent": parent,
        "parent_branch": parent_branch,
        "children": children,
    }


def path_to_root(
    node_id: int,
    tree: dict[str, Any],
) -> tuple[list[int], list[int]]:
    nodes = [node_id]
    branches: list[int] = []
    current = node_id
    while tree["parent"][current] is not None:
        branches.append(tree["parent_branch"][current])
        current = tree["parent"][current]
        nodes.append(current)
    nodes.reverse()
    branches.reverse()
    return nodes, branches


def carina_candidates(
    root_node_id: int,
    graph: dict[str, Any],
    radius_by_branch: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    node_by_id = graph["node_by_id"]
    branch_by_id = graph["branch_by_id"]
    adjacency = graph["adjacency"]
    tree = rooted_shortest_path_tree(root_node_id, graph)
    children = tree["children"]
    original_endpoint_ids = {
        node_id for node_id, edges in adjacency.items()
        if len(edges) == 1 and node_id != root_node_id
    }
    endpoint_count = {node_id: int(node_id in original_endpoint_ids) for node_id in node_by_id}
    endpoint_sum = {
        node_id: (
            np.asarray(node_by_id[node_id]["centroid_physical"], dtype=np.float64)
            if node_id in original_endpoint_ids
            else np.zeros(3, dtype=np.float64)
        )
        for node_id in node_by_id
    }
    subtree_length = {node_id: 0.0 for node_id in node_by_id}
    subtree_nodes = {node_id: 1 for node_id in node_by_id}
    for node_id in sorted(node_by_id, key=lambda value: tree["distance"][value], reverse=True):
        parent_id = tree["parent"][node_id]
        if parent_id is None:
            continue
        branch_id = tree["parent_branch"][node_id]
        endpoint_count[parent_id] += endpoint_count[node_id]
        endpoint_sum[parent_id] += endpoint_sum[node_id]
        subtree_length[parent_id] += subtree_length[node_id] + float(
            branch_by_id[branch_id]["length_mm"]
        )
        subtree_nodes[parent_id] += subtree_nodes[node_id]

    total_endpoints = len(original_endpoint_ids)
    all_radius_values = [
        radius_by_branch[branch_id]["p75"] for branch_id in branch_by_id
    ]
    root_point = np.asarray(
        node_by_id[root_node_id]["centroid_physical"], dtype=np.float64
    )
    candidates: list[dict[str, Any]] = []
    for node_id, node in node_by_id.items():
        child_records = [
            {
                "node_id": child,
                "branch_id": tree["parent_branch"][child],
                "endpoint_count": endpoint_count[child],
                "subtree_length_mm": subtree_length[child],
            }
            for child in children[node_id]
            if endpoint_count[child] > 0
        ]
        child_records.sort(
            key=lambda item: (
                -item["endpoint_count"],
                -item["subtree_length_mm"],
                item["node_id"],
            )
        )
        if len(child_records) < 2:
            continue
        first, second = child_records[:2]
        child_endpoint_sum = first["endpoint_count"] + second["endpoint_count"]
        coverage = child_endpoint_sum / total_endpoints
        balance = (
            2.0
            * min(first["endpoint_count"], second["endpoint_count"])
            / child_endpoint_sum
        )
        node_point = np.asarray(node["centroid_physical"], dtype=np.float64)
        direction_vectors: list[np.ndarray] = []
        for record in (first, second):
            centroid = endpoint_sum[record["node_id"]] / record["endpoint_count"]
            vector = centroid - node_point
            norm = float(np.linalg.norm(vector))
            direction_vectors.append(vector / norm if norm > 1e-9 else np.zeros(3))
        dot = float(np.clip(np.dot(direction_vectors[0], direction_vectors[1]), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(dot)))
        angle_score = float(np.clip(angle_deg / 90.0, 0.0, 1.0))
        lateral_divergence = float(
            np.clip(abs(direction_vectors[0][0] - direction_vectors[1][0]) / 2.0, 0.0, 1.0)
        )

        path_nodes, path_branches = path_to_root(node_id, tree)
        raw_path_length = float(
            sum(float(branch_by_id[branch_id]["length_mm"]) for branch_id in path_branches)
        )
        vertical_drop = float(root_point[2] - node_point[2])
        verticality = float(
            np.clip(vertical_drop / raw_path_length, 0.0, 1.0)
            if raw_path_length > 1e-9 else 0.0
        )
        straight_distance = float(np.linalg.norm(root_point - node_point))
        straightness = float(
            np.clip(straight_distance / raw_path_length, 0.0, 1.0)
            if raw_path_length > 1e-9 else 0.0
        )
        trachea_extent = float(np.clip(raw_path_length / 80.0, 0.0, 1.0))
        child_radius = float(
            np.mean(
                [
                    radius_by_branch[first["branch_id"]]["p75"],
                    radius_by_branch[second["branch_id"]]["p75"],
                ]
            )
        )
        caliber_score = percentile_rank(all_radius_values, child_radius)
        score = (
            0.30 * coverage
            + 0.22 * balance
            + 0.12 * lateral_divergence
            + 0.08 * angle_score
            + 0.12 * verticality
            + 0.08 * straightness
            + 0.05 * trachea_extent
            + 0.03 * caliber_score
        )
        candidates.append(
            {
                "node_id": node_id,
                "score": float(score),
                "physical_lps": node_point.tolist(),
                "endpoint_coverage": float(coverage),
                "subtree_balance": float(balance),
                "child_divergence_angle_deg": angle_deg,
                "lateral_divergence": lateral_divergence,
                "verticality": verticality,
                "straightness": straightness,
                "trachea_length_mm": raw_path_length,
                "trachea_node_ids": path_nodes,
                "trachea_branch_ids": path_branches,
                "main_child_node_ids": [first["node_id"], second["node_id"]],
                "main_child_branch_ids": [first["branch_id"], second["branch_id"]],
                "main_child_endpoint_counts": [
                    first["endpoint_count"],
                    second["endpoint_count"],
                ],
                "child_radius_p75_mean_mm": child_radius,
                "caliber_score": caliber_score,
                "tree_degree": len(children[node_id]) + int(tree["parent"][node_id] is not None),
                "original_multigraph_degree": len(adjacency[node_id]),
            }
        )
    if not candidates:
        raise DetectionError("no carina candidate has two downstream endpoint-bearing subtrees")
    candidates.sort(key=lambda item: (-item["score"], item["node_id"]))
    return candidates


def calibrated_confidence(candidates: list[dict[str, Any]]) -> float:
    top = float(candidates[0]["score"])
    second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
    margin_score = float(np.clip((top - second) / 0.15, 0.0, 1.0))
    return float(np.clip(0.75 * top + 0.25 * margin_score, 0.0, 1.0))


def weighted_trachea_radius(
    branch_ids: list[int],
    branch_by_id: dict[int, dict[str, Any]],
    radius_by_branch: dict[int, dict[str, float]],
) -> float:
    if not branch_ids:
        return 0.0
    values = np.asarray(
        [radius_by_branch[branch_id]["median"] for branch_id in branch_ids],
        dtype=np.float64,
    )
    weights = np.asarray(
        [max(float(branch_by_id[branch_id]["length_mm"]), EDGE_WEIGHT_FLOOR_MM) for branch_id in branch_ids],
        dtype=np.float64,
    )
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum())])


def orient_path_from_node(
    branch: dict[str, Any],
    physical: np.ndarray,
    node_id: int,
) -> np.ndarray:
    start = int(branch["start_node_id"])
    end = int(branch["end_node_id"])
    if node_id == start:
        return physical
    if node_id == end:
        return physical[::-1]
    raise DetectionError(f"branch {branch['id']} is not incident to node {node_id}")


def plot_qc(
    case_id: str,
    graph: dict[str, Any],
    physical_paths: dict[int, np.ndarray],
    root: dict[str, Any],
    carina: dict[str, Any],
    root_confidence: float,
    carina_confidence: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    projections = [
        (0, 2, "LPS X (+Left) mm", "LPS Z (+Superior) mm", "Coronal"),
        (1, 2, "LPS Y (+Posterior) mm", "LPS Z (+Superior) mm", "Sagittal"),
        (0, 1, "LPS X (+Left) mm", "LPS Y (+Posterior) mm", "Axial"),
    ]
    trachea_ids = set(carina["trachea_branch_ids"])
    child_ids = set(carina["main_child_branch_ids"])
    for axis, (horizontal, vertical, xlabel, ylabel, title) in zip(axes, projections):
        for branch_id, points in physical_paths.items():
            if branch_id in trachea_ids or branch_id in child_ids:
                continue
            axis.plot(
                points[:, horizontal],
                points[:, vertical],
                color="#7f8c8d",
                alpha=0.22,
                linewidth=0.55,
            )
        for branch_id in trachea_ids:
            points = physical_paths[branch_id]
            axis.plot(
                points[:, horizontal],
                points[:, vertical],
                color="#1565c0",
                linewidth=3.0,
                alpha=0.95,
            )
        child_colors = ["#2e7d32", "#8e24aa"]
        for branch_id, color in zip(carina["main_child_branch_ids"], child_colors):
            branch = graph["branch_by_id"][branch_id]
            points = orient_path_from_node(
                branch, physical_paths[branch_id], carina["node_id"]
            )
            axis.plot(
                points[:, horizontal],
                points[:, vertical],
                color=color,
                linewidth=3.0,
                alpha=0.95,
            )
        root_point = np.asarray(root["physical_lps"])
        carina_point = np.asarray(carina["physical_lps"])
        axis.scatter(
            root_point[horizontal],
            root_point[vertical],
            marker="*",
            s=180,
            color="#d32f2f",
            edgecolor="black",
            linewidth=0.7,
            zorder=10,
            label=f"Root node {root['node_id']}",
        )
        axis.scatter(
            carina_point[horizontal],
            carina_point[vertical],
            marker="D",
            s=95,
            color="#f9a825",
            edgecolor="black",
            linewidth=0.7,
            zorder=10,
            label=f"Carina node {carina['node_id']}",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.15)
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        f"{case_id} | root confidence={root_confidence:.3f} "
        f"| carina confidence={carina_confidence:.3f}\n"
        f"Blue: detected trachea; green/purple: two dominant distal trunks",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    temporary = output_path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, output_path)


def analyze_case(
    case_id: str,
    clean_root: Path,
    label_root: Path,
    qc_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir = clean_root / case_id
    graph = load_graph(case_dir, case_id)
    physical_paths, radius_by_branch = load_paths_and_radius(
        case_id,
        case_dir,
        label_root / f"{case_id}.nii.gz",
        graph["nodes"],
        graph["branches"],
    )
    root_candidates = endpoint_root_candidates(graph, radius_by_branch)
    root = root_candidates[0]
    root_confidence = calibrated_confidence(root_candidates)
    carina_ranked = carina_candidates(root["node_id"], graph, radius_by_branch)
    carina = carina_ranked[0]
    carina_confidence = calibrated_confidence(carina_ranked)
    branch_by_id = graph["branch_by_id"]
    trachea_radius = weighted_trachea_radius(
        carina["trachea_branch_ids"], branch_by_id, radius_by_branch
    )

    review_reasons: list[str] = []
    if confidence_label(root_confidence) == "low":
        review_reasons.append("low_root_confidence")
    if confidence_label(carina_confidence) == "low":
        review_reasons.append("low_carina_confidence")
    if carina["endpoint_coverage"] < 0.75:
        review_reasons.append("carina_low_distal_endpoint_coverage")
    if carina["subtree_balance"] < 0.35:
        review_reasons.append("carina_unbalanced_main_subtrees")
    if carina["verticality"] < 0.50:
        review_reasons.append("trachea_not_sufficiently_inferior_directed")
    if carina["trachea_length_mm"] <= 0:
        review_reasons.append("empty_trachea_path")

    qc_path = qc_root / f"{case_id}_root_carina_qc.png"
    plot_qc(
        case_id,
        graph,
        physical_paths,
        root,
        carina,
        root_confidence,
        carina_confidence,
        qc_path,
    )
    root_point = root["physical_lps"]
    carina_point = carina["physical_lps"]
    row = {
        "case_id": case_id,
        "root_node_id": root["node_id"],
        "root_lps_x_mm": root_point[0],
        "root_lps_y_mm": root_point[1],
        "root_lps_z_mm": root_point[2],
        "root_incident_branch_id": root["branch_id"],
        "root_score": root["score"],
        "root_confidence": root_confidence,
        "root_confidence_label": confidence_label(root_confidence),
        "carina_node_id": carina["node_id"],
        "carina_lps_x_mm": carina_point[0],
        "carina_lps_y_mm": carina_point[1],
        "carina_lps_z_mm": carina_point[2],
        "carina_score": carina["score"],
        "carina_confidence": carina_confidence,
        "carina_confidence_label": confidence_label(carina_confidence),
        "trachea_node_ids": json.dumps(carina["trachea_node_ids"], separators=(",", ":")),
        "trachea_branch_ids": json.dumps(carina["trachea_branch_ids"], separators=(",", ":")),
        "trachea_length_mm": carina["trachea_length_mm"],
        "trachea_median_radius_mm": trachea_radius,
        "trachea_verticality": carina["verticality"],
        "trachea_straightness": carina["straightness"],
        "carina_endpoint_coverage": carina["endpoint_coverage"],
        "carina_subtree_balance": carina["subtree_balance"],
        "carina_child_divergence_angle_deg": carina["child_divergence_angle_deg"],
        "carina_lateral_divergence": carina["lateral_divergence"],
        "carina_main_child_branch_ids": json.dumps(
            carina["main_child_branch_ids"], separators=(",", ":")
        ),
        "residual_cycle_rank_unchanged": graph["cycle_rank"],
        "needs_manual_review": bool(review_reasons),
        "review_reasons": ";".join(review_reasons),
        "qc_png": str(qc_path),
    }
    details = {
        "case_id": case_id,
        "coordinate_system": "LPS",
        "root_candidate": root,
        "root_confidence": root_confidence,
        "root_confidence_label": confidence_label(root_confidence),
        "top_root_candidates": root_candidates[:10],
        "carina_candidate": carina,
        "carina_confidence": carina_confidence,
        "carina_confidence_label": confidence_label(carina_confidence),
        "top_carina_candidates": carina_ranked[:10],
        "trachea_median_radius_mm": trachea_radius,
        "needs_manual_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "residual_cycle_rank_unchanged": graph["cycle_rank"],
    }
    return row, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CLEAN_ROOT / "trachea_root_carina_candidates.csv",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=DEFAULT_CLEAN_ROOT / "trachea_root_carina_candidate_details.json",
    )
    parser.add_argument(
        "--qc-root",
        type=Path,
        default=DEFAULT_CLEAN_ROOT / "root_carina_qc",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_root = args.clean_root.resolve()
    label_root = args.label_root.resolve()
    cases = list(dict.fromkeys(args.cases))
    output = args.output.resolve()
    details_output = args.details_output.resolve()
    qc_root = args.qc_root.resolve()
    expected_outputs = [output, details_output] + [
        qc_root / f"{case_id}_root_carina_qc.png" for case_id in cases
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        raise DetectionError(
            "refusing to overwrite existing Stage-3 outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    qc_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "stage": "Stage 3 trachea root / carina candidate detection",
        "coordinate_system": "LPS",
        "case_count": len(cases),
        "cases": [],
    }
    for case_id in cases:
        row, case_details = analyze_case(
            case_id, clean_root, label_root, qc_root
        )
        rows.append(row)
        details["cases"].append(case_details)
        print(
            f"{case_id}: root={row['root_node_id']} "
            f"({row['root_confidence_label']} {row['root_confidence']:.3f}), "
            f"carina={row['carina_node_id']} "
            f"({row['carina_confidence_label']} {row['carina_confidence']:.3f}), "
            f"review={row['needs_manual_review']}",
            flush=True,
        )

    temporary_csv = output.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, output)
    temporary_json = details_output.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(details, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_json, details_output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    if len(written) != len(cases):
        raise DetectionError("candidate CSV row-count mismatch")
    if any(not Path(row["qc_png"]).is_file() for row in written):
        raise DetectionError("a QC image is missing")
    print(f"wrote {output}")
    print(f"wrote {details_output}")
    print(f"wrote {qc_root}: {len(cases)} QC images")
    print("No graph or residual cycle was modified.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, TypeError, ValueError, DetectionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
