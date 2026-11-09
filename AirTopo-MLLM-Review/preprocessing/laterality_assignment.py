#!/usr/bin/env python3
"""Stage-4 left/right subtree assignment for the five validated test cases.

This script is deliberately read-only with respect to nodes.json, branches.json,
branch arrays, and residual cycles.  It writes a separate side_assignment.json
per case plus a five-case summary and projection QC images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import itertools
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
from matplotlib.lines import Line2D
import numpy as np


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_CLEAN_ROOT = DATA_ROOT / "ATM22_processed" / "topology_clean"
DEFAULT_CASES = ()
EDGE_WEIGHT_FLOOR_MM = 0.4

SUMMARY_FIELDS = [
    "case_id",
    "root_node_id",
    "carina_node_id",
    "left_main_branch_id",
    "right_main_branch_id",
    "left_subtree_node_count",
    "right_subtree_node_count",
    "left_subtree_branch_count",
    "right_subtree_branch_count",
    "left_endpoint_count",
    "right_endpoint_count",
    "left_confidence",
    "right_confidence",
    "confidence_label",
    "left_centroid_lps_x_mm",
    "right_centroid_lps_x_mm",
    "left_endpoint_median_lps_x_mm",
    "right_endpoint_median_lps_x_mm",
    "unknown_nodes",
    "unknown_branches",
    "assigned_nontrachea_node_fraction",
    "assigned_nontrachea_branch_fraction",
    "both_subtrees_start_at_carina",
    "left_right_disjoint",
    "left_subtree_connected",
    "right_subtree_connected",
    "graph_connectivity_unchanged",
    "node_count_unchanged",
    "branch_count_unchanged",
    "residual_cycle_rank_unchanged",
    "needs_manual_review",
    "review_reasons",
    "assignment_json",
    "qc_png",
]


class AssignmentError(RuntimeError):
    """Raised when graph or Stage-3 candidate validation fails."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssignmentError(f"{path}: expected JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def graph_component_count(
    node_by_id: dict[int, dict[str, Any]],
    branch_by_id: dict[int, dict[str, Any]],
) -> int:
    adjacency = {node_id: set() for node_id in node_by_id}
    for branch in branch_by_id.values():
        start = int(branch["start_node_id"])
        end = int(branch["end_node_id"])
        if start == end:
            raise AssignmentError(f"self-loop remains: branch {branch['id']}")
        if start not in adjacency or end not in adjacency:
            raise AssignmentError(f"branch {branch['id']} references a missing node")
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
    nodes_path = case_dir / "nodes.json"
    branches_path = case_dir / "branches.json"
    nodes_doc = load_json(nodes_path)
    branches_doc = load_json(branches_path)
    stats = load_json(case_dir / "stats.json")
    if nodes_doc.get("case_id") != case_id or branches_doc.get("case_id") != case_id:
        raise AssignmentError(f"{case_id}: case ID mismatch")
    if nodes_doc.get("coordinate_system") != "LPS" or branches_doc.get("coordinate_system") != "LPS":
        raise AssignmentError(f"{case_id}: expected LPS graph coordinates")
    nodes = nodes_doc["nodes"]
    branches = branches_doc["branches"]
    node_by_id = {int(node["id"]): node for node in nodes}
    branch_by_id = {int(branch["id"]): branch for branch in branches}
    if len(node_by_id) != len(nodes) or len(branch_by_id) != len(branches):
        raise AssignmentError(f"{case_id}: duplicate node or branch IDs")
    components = graph_component_count(node_by_id, branch_by_id)
    cycle_rank = len(branches) - len(nodes) + components
    if components != 1 or not bool(stats["clean_is_connected"]):
        raise AssignmentError(f"{case_id}: clean graph is not connected")
    if cycle_rank != int(stats["residual_cycle_after"]):
        raise AssignmentError(f"{case_id}: residual cycle rank mismatch")

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
        "components": components,
        "cycle_rank": cycle_rank,
        "source_hashes": {
            "nodes.json": file_sha256(nodes_path),
            "branches.json": file_sha256(branches_path),
        },
    }


def rooted_shortest_path_tree(root_node_id: int, graph: dict[str, Any]) -> dict[str, Any]:
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
            proposed = current_distance + max(float(length), EDGE_WEIGHT_FLOOR_MM)
            current_key = (
                parent.get(neighbor, sys.maxsize),
                parent_branch.get(neighbor, sys.maxsize),
            )
            proposed_key = (node_id, branch_id)
            if (
                proposed < distance[neighbor] - 1e-12
                or (
                    abs(proposed - distance[neighbor]) <= 1e-12
                    and proposed_key < current_key
                )
            ):
                distance[neighbor] = proposed
                parent[neighbor] = node_id
                parent_branch[neighbor] = branch_id
                heapq.heappush(queue, (proposed, neighbor))
    if len(parent) != len(adjacency):
        raise AssignmentError("shortest-path tree did not reach every node")
    children = {node_id: [] for node_id in adjacency}
    for node_id, parent_id in parent.items():
        if parent_id is not None:
            children[parent_id].append(node_id)
    for child_ids in children.values():
        child_ids.sort()
    return {
        "distance": distance,
        "parent": parent,
        "parent_branch": parent_branch,
        "children": children,
    }


def descendants(seed: int, children: dict[int, list[int]]) -> set[int]:
    result: set[int] = set()
    stack = [seed]
    while stack:
        node_id = stack.pop()
        if node_id in result:
            continue
        result.add(node_id)
        stack.extend(children[node_id])
    return result


def orient_branch_path(
    path: np.ndarray,
    source_point: np.ndarray,
) -> np.ndarray:
    if np.linalg.norm(path[-1] - source_point) < np.linalg.norm(path[0] - source_point):
        return path[::-1]
    return path


def initial_branch_vector(
    branch_id: int,
    source_node_id: int,
    graph: dict[str, Any],
    physical_paths: dict[int, np.ndarray],
) -> np.ndarray:
    source = np.asarray(
        graph["node_by_id"][source_node_id]["centroid_physical"], dtype=np.float64
    )
    path = orient_branch_path(physical_paths[branch_id], source)
    distances = np.linalg.norm(path - path[0], axis=1)
    eligible = np.flatnonzero(distances <= 12.0)
    index = int(eligible[-1]) if len(eligible) else min(len(path) - 1, 1)
    if index == 0 and len(path) > 1:
        index = 1
    vector = path[index] - path[0]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        branch = graph["branch_by_id"][branch_id]
        other = (
            int(branch["end_node_id"])
            if int(branch["start_node_id"]) == source_node_id
            else int(branch["start_node_id"])
        )
        vector = np.asarray(
            graph["node_by_id"][other]["centroid_physical"], dtype=np.float64
        ) - source
        norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def subtree_record(
    child: int,
    carina: int,
    tree: dict[str, Any],
    graph: dict[str, Any],
    physical_paths: dict[int, np.ndarray],
) -> dict[str, Any]:
    node_ids = descendants(child, tree["children"])
    branch_ids = {tree["parent_branch"][node_id] for node_id in node_ids}
    endpoint_ids = {
        node_id for node_id in node_ids if len(graph["adjacency"][node_id]) == 1
    }
    points = np.asarray(
        [graph["node_by_id"][node_id]["centroid_physical"] for node_id in node_ids],
        dtype=np.float64,
    )
    endpoint_points = np.asarray(
        [graph["node_by_id"][node_id]["centroid_physical"] for node_id in endpoint_ids],
        dtype=np.float64,
    )
    if len(endpoint_points) == 0:
        endpoint_points = points
    endpoint_distances = np.asarray([tree["distance"][node_id] for node_id in endpoint_ids])
    if len(endpoint_ids):
        threshold = float(np.median(endpoint_distances))
        distal_ids = [
            node_id for node_id in endpoint_ids if tree["distance"][node_id] >= threshold
        ]
        distal_points = np.asarray(
            [graph["node_by_id"][node_id]["centroid_physical"] for node_id in distal_ids],
            dtype=np.float64,
        )
    else:
        distal_points = points
    branch_id = tree["parent_branch"][child]
    return {
        "child_node_id": child,
        "main_branch_id": branch_id,
        "node_ids": node_ids,
        "tree_branch_ids": branch_ids,
        "endpoint_ids": endpoint_ids,
        "node_count": len(node_ids),
        "tree_branch_count": len(branch_ids),
        "endpoint_count": len(endpoint_ids),
        "tree_branch_length_mm": float(sum(
            float(graph["branch_by_id"][bid]["length_mm"]) for bid in branch_ids
        )),
        "centroid_lps": points.mean(axis=0),
        "endpoint_median_lps": np.median(endpoint_points, axis=0),
        "endpoint_q25_x_mm": float(np.percentile(endpoint_points[:, 0], 25)),
        "endpoint_q75_x_mm": float(np.percentile(endpoint_points[:, 0], 75)),
        "distal_endpoint_median_lps": np.median(distal_points, axis=0),
        "initial_vector": initial_branch_vector(
            branch_id, carina, graph, physical_paths
        ),
    }


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def choose_main_subtrees(
    carina: int,
    trachea_nodes: list[int],
    tree: dict[str, Any],
    graph: dict[str, Any],
    physical_paths: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(trachea_nodes) < 2 or trachea_nodes[-1] != carina:
        raise AssignmentError("invalid root-to-carina trachea path")
    proximal = np.asarray(
        graph["node_by_id"][trachea_nodes[-2]]["centroid_physical"],
        dtype=np.float64,
    )
    carina_point = np.asarray(
        graph["node_by_id"][carina]["centroid_physical"], dtype=np.float64
    )
    trachea_distal_vector = carina_point - proximal
    trachea_distal_vector /= max(float(np.linalg.norm(trachea_distal_vector)), 1e-12)
    child_records = [
        subtree_record(child, carina, tree, graph, physical_paths)
        for child in tree["children"][carina]
    ]
    if len(child_records) < 2:
        raise AssignmentError("carina has fewer than two downstream tree children")
    total_endpoints = sum(record["endpoint_count"] for record in child_records)
    total_length = sum(record["tree_branch_length_mm"] for record in child_records)
    total_nodes = sum(record["node_count"] for record in child_records)
    pair_candidates: list[tuple[float, tuple[int, int], dict[str, float]]] = []
    for i, j in itertools.combinations(range(len(child_records)), 2):
        first, second = child_records[i], child_records[j]
        endpoint_coverage = (
            (first["endpoint_count"] + second["endpoint_count"]) / total_endpoints
            if total_endpoints else 0.0
        )
        length_coverage = (
            (first["tree_branch_length_mm"] + second["tree_branch_length_mm"]) / total_length
            if total_length else 0.0
        )
        node_coverage = (
            (first["node_count"] + second["node_count"]) / total_nodes
            if total_nodes else 0.0
        )
        larger = max(first["endpoint_count"], second["endpoint_count"], 1)
        balance = min(first["endpoint_count"], second["endpoint_count"]) / larger
        divergence = angle_degrees(first["initial_vector"], second["initial_vector"])
        divergence_score = float(np.clip(divergence / 75.0, 0.0, 1.0))
        first_trachea_angle = angle_degrees(trachea_distal_vector, first["initial_vector"])
        second_trachea_angle = angle_degrees(trachea_distal_vector, second["initial_vector"])
        distal_forward_score = 0.5 * (
            max(float(np.dot(trachea_distal_vector, first["initial_vector"])), 0.0)
            + max(float(np.dot(trachea_distal_vector, second["initial_vector"])), 0.0)
        )
        lateral_score = float(np.clip(
            abs(first["initial_vector"][0] - second["initial_vector"][0]) / 1.2,
            0.0,
            1.0,
        ))
        score = (
            0.32 * endpoint_coverage
            + 0.23 * length_coverage
            + 0.10 * node_coverage
            + 0.15 * balance
            + 0.08 * divergence_score
            + 0.05 * lateral_score
            + 0.07 * distal_forward_score
        )
        pair_candidates.append((score, (i, j), {
            "selection_score": score,
            "endpoint_coverage": endpoint_coverage,
            "length_coverage": length_coverage,
            "node_coverage": node_coverage,
            "endpoint_balance": balance,
            "divergence_angle_deg": divergence,
            "first_angle_from_trachea_deg": first_trachea_angle,
            "second_angle_from_trachea_deg": second_trachea_angle,
            "distal_forward_score": distal_forward_score,
        }))
    pair_candidates.sort(
        key=lambda item: (
            -item[0],
            child_records[item[1][0]]["main_branch_id"],
            child_records[item[1][1]]["main_branch_id"],
        )
    )
    _, (i, j), metrics = pair_candidates[0]
    return [child_records[i], child_records[j]], metrics


def assign_left_right(
    first: dict[str, Any],
    second: dict[str, Any],
    carina_point: np.ndarray,
    main_pair_coverage: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    def lateral_signature(record: dict[str, Any]) -> float:
        return float(
            0.30 * record["centroid_lps"][0]
            + 0.30 * record["endpoint_median_lps"][0]
            + 0.25 * record["distal_endpoint_median_lps"][0]
            + 0.15 * (carina_point[0] + 40.0 * record["initial_vector"][0])
        )

    first_score = lateral_signature(first)
    second_score = lateral_signature(second)
    if first_score >= second_score:
        left, right = first, second
        left_score, right_score = first_score, second_score
    else:
        left, right = second, first
        left_score, right_score = second_score, first_score

    comparisons = [
        left["centroid_lps"][0] > right["centroid_lps"][0],
        left["endpoint_median_lps"][0] > right["endpoint_median_lps"][0],
        left["distal_endpoint_median_lps"][0] > right["distal_endpoint_median_lps"][0],
        left["initial_vector"][0] > right["initial_vector"][0],
    ]
    agreement = float(sum(comparisons) / len(comparisons))
    pooled_spread = max(
        10.0,
        0.5 * (
            left["endpoint_q75_x_mm"] - left["endpoint_q25_x_mm"]
            + right["endpoint_q75_x_mm"] - right["endpoint_q25_x_mm"]
        ),
    )
    separation = float(left_score - right_score)
    separation_score = float(1.0 - math.exp(-max(separation, 0.0) / pooled_spread))
    confidence = float(np.clip(
        0.55 * separation_score + 0.25 * agreement + 0.20 * main_pair_coverage,
        0.0,
        1.0,
    ))
    return left, right, {
        "left_lateral_signature_mm": left_score,
        "right_lateral_signature_mm": right_score,
        "lateral_signature_separation_mm": separation,
        "lateral_evidence_agreement": agreement,
        "pooled_endpoint_x_spread_mm": pooled_spread,
        "confidence": confidence,
    }


def side_connected(
    side: str,
    main_branch_id: int,
    carina: int,
    node_side: dict[int, str],
    branch_side: dict[int, str],
    graph: dict[str, Any],
) -> bool:
    branch = graph["branch_by_id"][main_branch_id]
    seed = (
        int(branch["end_node_id"])
        if int(branch["start_node_id"]) == carina
        else int(branch["start_node_id"])
    )
    target_nodes = {node_id for node_id, value in node_side.items() if value == side}
    if seed not in target_nodes:
        return False
    adjacency = {node_id: set() for node_id in target_nodes}
    for branch_id, value in branch_side.items():
        if value != side:
            continue
        item = graph["branch_by_id"][branch_id]
        start, end = int(item["start_node_id"]), int(item["end_node_id"])
        if start == carina:
            start = seed
        if end == carina:
            end = seed
        if start in adjacency and end in adjacency:
            adjacency[start].add(end)
            adjacency[end].add(start)
    seen = set()
    stack = [seed]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(adjacency[node_id] - seen)
    return seen == target_nodes


def confidence_label(value: float) -> str:
    if value >= 0.80:
        return "high"
    if value >= 0.65:
        return "medium"
    return "low"


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def plot_qc(
    case_id: str,
    graph: dict[str, Any],
    physical_paths: dict[int, np.ndarray],
    node_side: dict[int, str],
    branch_side: dict[int, str],
    root: int,
    carina: int,
    confidence: float,
    output_path: Path,
) -> None:
    colors = {
        "trachea": "#1f77b4",
        "left": "#d62728",
        "right": "#2ca02c",
        "unknown": "#8c8c8c",
    }
    widths = {"trachea": 2.8, "left": 1.45, "right": 1.45, "unknown": 0.7}
    alphas = {"trachea": 1.0, "left": 0.82, "right": 0.82, "unknown": 0.35}
    projections = [
        (0, 2, "Coronal", "LPS X (+Left) mm", "LPS Z (+Superior) mm"),
        (1, 2, "Sagittal", "LPS Y (+Posterior) mm", "LPS Z (+Superior) mm"),
        (0, 1, "Axial", "LPS X (+Left) mm", "LPS Y (+Posterior) mm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    draw_order = ["unknown", "right", "left", "trachea"]
    for ax, (horizontal, vertical, title, xlabel, ylabel) in zip(axes, projections):
        for side in draw_order:
            for branch_id, value in branch_side.items():
                if value != side:
                    continue
                path = physical_paths[branch_id]
                ax.plot(
                    path[:, horizontal], path[:, vertical],
                    color=colors[side], linewidth=widths[side], alpha=alphas[side],
                )
        root_point = np.asarray(graph["node_by_id"][root]["centroid_physical"])
        carina_point = np.asarray(graph["node_by_id"][carina]["centroid_physical"])
        ax.scatter(root_point[horizontal], root_point[vertical], marker="*", s=95,
                   color="black", edgecolor="white", linewidth=0.6, zorder=10)
        ax.scatter(carina_point[horizontal], carina_point[vertical], marker="D", s=55,
                   color="#ffbf00", edgecolor="black", linewidth=0.6, zorder=10)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
    counts = Counter(branch_side.values())
    fig.suptitle(
        f"{case_id} | L/R confidence={confidence:.3f} | "
        f"branches: trachea={counts['trachea']}, left={counts['left']}, "
        f"right={counts['right']}, unknown={counts['unknown']}"
    )
    handles = [
        Line2D([0], [0], color=colors[side], lw=3, label=side)
        for side in ["trachea", "left", "right", "unknown"]
    ] + [
        Line2D([0], [0], marker="*", color="black", lw=0, markersize=10, label="root"),
        Line2D([0], [0], marker="D", markerfacecolor="#ffbf00", markeredgecolor="black",
               color="none", lw=0, markersize=7, label="carina"),
    ]
    axes[2].legend(handles=handles, loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, output_path)


def analyze_case(
    case_id: str,
    clean_root: Path,
    stage3: dict[str, Any],
    qc_dir: Path,
) -> dict[str, Any]:
    case_dir = clean_root / case_id
    graph = load_graph(case_dir, case_id)
    physical_paths = {
        branch_id: np.load(case_dir / branch["physical_file"], allow_pickle=False).astype(
            np.float64, copy=False
        )
        for branch_id, branch in graph["branch_by_id"].items()
    }
    root = int(stage3["root_candidate"]["node_id"])
    carina = int(stage3["carina_candidate"]["node_id"])
    trachea_nodes = [int(value) for value in stage3["carina_candidate"]["trachea_node_ids"]]
    trachea_branches = [int(value) for value in stage3["carina_candidate"]["trachea_branch_ids"]]
    if trachea_nodes[0] != root or trachea_nodes[-1] != carina:
        raise AssignmentError(f"{case_id}: invalid Stage-3 trachea node path")
    tree = rooted_shortest_path_tree(root, graph)
    selected, pair_metrics = choose_main_subtrees(
        carina, trachea_nodes, tree, graph, physical_paths
    )
    selected_branch_ids = {record["main_branch_id"] for record in selected}
    stage3_branch_ids = {
        int(value) for value in stage3["carina_candidate"]["main_child_branch_ids"]
    }
    stage3_pair_agrees = selected_branch_ids == stage3_branch_ids
    carina_point = np.asarray(
        graph["node_by_id"][carina]["centroid_physical"], dtype=np.float64
    )
    left, right, lateral_metrics = assign_left_right(
        selected[0], selected[1], carina_point, pair_metrics["endpoint_coverage"]
    )

    node_side = {node_id: "unknown" for node_id in graph["node_by_id"]}
    node_reason = {node_id: "outside_two_dominant_carina_subtrees" for node_id in node_side}
    for node_id in trachea_nodes:
        node_side[node_id] = "trachea"
        node_reason[node_id] = "stage3_root_to_carina_path"
    for side, record in [("left", left), ("right", right)]:
        for node_id in record["node_ids"]:
            if node_side[node_id] != "unknown":
                raise AssignmentError(f"{case_id}: overlapping {side} and trachea node {node_id}")
            node_side[node_id] = side
            node_reason[node_id] = f"descendant_of_{side}_main_branch"

    branch_side = {branch_id: "unknown" for branch_id in graph["branch_by_id"]}
    branch_reason = {
        branch_id: "non_tree_or_ambiguous_structure" for branch_id in branch_side
    }
    for branch_id in trachea_branches:
        branch_side[branch_id] = "trachea"
        branch_reason[branch_id] = "stage3_root_to_carina_path"
    for side, record in [("left", left), ("right", right)]:
        for branch_id in record["tree_branch_ids"]:
            if branch_side[branch_id] != "unknown":
                raise AssignmentError(f"{case_id}: overlapping tree branch {branch_id}")
            branch_side[branch_id] = side
            branch_reason[branch_id] = f"shortest_path_tree_edge_in_{side}_subtree"
    for branch_id, branch in graph["branch_by_id"].items():
        if branch_side[branch_id] != "unknown":
            continue
        start = int(branch["start_node_id"])
        end = int(branch["end_node_id"])
        if node_side[start] == node_side[end] and node_side[start] in {"left", "right"}:
            branch_side[branch_id] = node_side[start]
            branch_reason[branch_id] = "same_side_residual_cycle_edge"

    both_start = True
    for record in [left, right]:
        branch = graph["branch_by_id"][record["main_branch_id"]]
        both_start &= carina in {int(branch["start_node_id"]), int(branch["end_node_id"])}
    left_nodes = {node_id for node_id, side in node_side.items() if side == "left"}
    right_nodes = {node_id for node_id, side in node_side.items() if side == "right"}
    left_branches = {branch_id for branch_id, side in branch_side.items() if side == "left"}
    right_branches = {branch_id for branch_id, side in branch_side.items() if side == "right"}
    unknown_nodes = sorted(node_id for node_id, side in node_side.items() if side == "unknown")
    unknown_branches = sorted(
        branch_id for branch_id, side in branch_side.items() if side == "unknown"
    )
    trachea_node_set = set(trachea_nodes)
    nontrachea_node_denominator = len(graph["nodes"]) - len(trachea_node_set)
    nontrachea_branch_denominator = len(graph["branches"]) - len(trachea_branches)
    assigned_node_fraction = (
        (len(left_nodes) + len(right_nodes)) / nontrachea_node_denominator
        if nontrachea_node_denominator else 1.0
    )
    assigned_branch_fraction = (
        (len(left_branches) + len(right_branches)) / nontrachea_branch_denominator
        if nontrachea_branch_denominator else 1.0
    )
    left_connected = side_connected(
        "left", left["main_branch_id"], carina, node_side, branch_side, graph
    )
    right_connected = side_connected(
        "right", right["main_branch_id"], carina, node_side, branch_side, graph
    )

    current_hashes = {
        "nodes.json": file_sha256(case_dir / "nodes.json"),
        "branches.json": file_sha256(case_dir / "branches.json"),
    }
    source_unchanged = current_hashes == graph["source_hashes"]
    confidence = lateral_metrics["confidence"]
    review_reasons: list[str] = []
    if confidence < 0.65:
        review_reasons.append("low_left_right_confidence")
    if lateral_metrics["lateral_evidence_agreement"] < 0.75:
        review_reasons.append("lateral_evidence_disagreement")
    if not stage3_pair_agrees:
        review_reasons.append("main_pair_disagrees_with_stage3")
    if not both_start:
        review_reasons.append("main_subtree_not_incident_to_carina")
    if not left_connected or not right_connected:
        review_reasons.append("assigned_subtree_not_connected")
    if assigned_node_fraction < 0.95 or assigned_branch_fraction < 0.95:
        review_reasons.append("substantial_unknown_structure")
    if not source_unchanged:
        raise AssignmentError(f"{case_id}: source graph changed during assignment")

    assignment_path = case_dir / "side_assignment.json"
    qc_path = qc_dir / f"{case_id}_left_right_qc.png"
    document = {
        "case_id": case_id,
        "stage": "left_right_subtree_assignment_candidate",
        "coordinate_system": "LPS",
        "lps_axis_note": "+X=Left, -X=Right",
        "source_graph_modified": False,
        "root_node_id": root,
        "carina_node_id": carina,
        "trachea_node_ids": trachea_nodes,
        "trachea_branch_ids": trachea_branches,
        "left_main_branch_id": left["main_branch_id"],
        "right_main_branch_id": right["main_branch_id"],
        "left_main_child_node_id": left["child_node_id"],
        "right_main_child_node_id": right["child_node_id"],
        "main_subtree_selection": {
            **pair_metrics,
            "stage3_pair_agrees": stage3_pair_agrees,
        },
        "lateral_assignment_evidence": {
            **lateral_metrics,
            "left_centroid_lps": left["centroid_lps"].tolist(),
            "right_centroid_lps": right["centroid_lps"].tolist(),
            "left_endpoint_median_lps": left["endpoint_median_lps"].tolist(),
            "right_endpoint_median_lps": right["endpoint_median_lps"].tolist(),
            "left_distal_endpoint_median_lps": left["distal_endpoint_median_lps"].tolist(),
            "right_distal_endpoint_median_lps": right["distal_endpoint_median_lps"].tolist(),
            "left_initial_direction_lps": left["initial_vector"].tolist(),
            "right_initial_direction_lps": right["initial_vector"].tolist(),
        },
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "counts": {
            "trachea_nodes": sum(side == "trachea" for side in node_side.values()),
            "left_nodes": len(left_nodes),
            "right_nodes": len(right_nodes),
            "unknown_nodes": len(unknown_nodes),
            "trachea_branches": sum(side == "trachea" for side in branch_side.values()),
            "left_branches": len(left_branches),
            "right_branches": len(right_branches),
            "unknown_branches": len(unknown_branches),
            "left_endpoints": sum(len(graph["adjacency"][node_id]) == 1 for node_id in left_nodes),
            "right_endpoints": sum(len(graph["adjacency"][node_id]) == 1 for node_id in right_nodes),
        },
        "unknown_node_ids": unknown_nodes,
        "unknown_branch_ids": unknown_branches,
        "node_assignments": [
            {"node_id": node_id, "side": node_side[node_id], "reason": node_reason[node_id]}
            for node_id in sorted(node_side)
        ],
        "branch_assignments": [
            {"branch_id": branch_id, "side": branch_side[branch_id], "reason": branch_reason[branch_id]}
            for branch_id in sorted(branch_side)
        ],
        "validation": {
            "both_subtrees_start_at_carina": both_start,
            "left_right_disjoint": not bool(left_nodes & right_nodes or left_branches & right_branches),
            "left_subtree_connected": left_connected,
            "right_subtree_connected": right_connected,
            "assigned_nontrachea_node_fraction": assigned_node_fraction,
            "assigned_nontrachea_branch_fraction": assigned_branch_fraction,
            "graph_components_before": graph["components"],
            "graph_components_after": graph["components"],
            "node_count_before": len(graph["nodes"]),
            "node_count_after": len(graph["nodes"]),
            "branch_count_before": len(graph["branches"]),
            "branch_count_after": len(graph["branches"]),
            "residual_cycle_rank_before": graph["cycle_rank"],
            "residual_cycle_rank_after": graph["cycle_rank"],
            "source_graph_sha256_unchanged": source_unchanged,
        },
        "needs_manual_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }
    write_json_atomic(assignment_path, document)
    plot_qc(
        case_id, graph, physical_paths, node_side, branch_side,
        root, carina, confidence, qc_path,
    )
    counts = document["counts"]
    validation = document["validation"]
    return {
        "case_id": case_id,
        "root_node_id": root,
        "carina_node_id": carina,
        "left_main_branch_id": left["main_branch_id"],
        "right_main_branch_id": right["main_branch_id"],
        "left_subtree_node_count": counts["left_nodes"],
        "right_subtree_node_count": counts["right_nodes"],
        "left_subtree_branch_count": counts["left_branches"],
        "right_subtree_branch_count": counts["right_branches"],
        "left_endpoint_count": counts["left_endpoints"],
        "right_endpoint_count": counts["right_endpoints"],
        "left_confidence": confidence,
        "right_confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "left_centroid_lps_x_mm": float(left["centroid_lps"][0]),
        "right_centroid_lps_x_mm": float(right["centroid_lps"][0]),
        "left_endpoint_median_lps_x_mm": float(left["endpoint_median_lps"][0]),
        "right_endpoint_median_lps_x_mm": float(right["endpoint_median_lps"][0]),
        "unknown_nodes": counts["unknown_nodes"],
        "unknown_branches": counts["unknown_branches"],
        "assigned_nontrachea_node_fraction": assigned_node_fraction,
        "assigned_nontrachea_branch_fraction": assigned_branch_fraction,
        "both_subtrees_start_at_carina": both_start,
        "left_right_disjoint": validation["left_right_disjoint"],
        "left_subtree_connected": left_connected,
        "right_subtree_connected": right_connected,
        "graph_connectivity_unchanged": source_unchanged and graph["components"] == 1,
        "node_count_unchanged": True,
        "branch_count_unchanged": True,
        "residual_cycle_rank_unchanged": graph["cycle_rank"],
        "needs_manual_review": bool(review_reasons),
        "review_reasons": ";".join(review_reasons),
        "assignment_json": str(assignment_path.resolve()),
        "qc_png": str(qc_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign trachea/left/right/unknown labels for selected cases."
    )
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--cases", nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_root = args.clean_root.resolve()
    cases = list(dict.fromkeys(args.cases))
    stage3_doc = load_json(clean_root / "trachea_root_carina_candidate_details.json")
    if stage3_doc.get("coordinate_system") != "LPS":
        raise AssignmentError("Stage-3 candidate coordinates are not LPS")
    stage3_by_case = {item["case_id"]: item for item in stage3_doc["cases"]}
    qc_dir = clean_root / "left_right_qc"
    rows = [
        analyze_case(case_id, clean_root, stage3_by_case[case_id], qc_dir)
        for case_id in cases
    ]
    output_csv = clean_root / "left_right_assignment_candidates.csv"
    temporary = output_csv.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_csv)
    for row in rows:
        print(
            f"{row['case_id']}: left_branch={row['left_main_branch_id']} "
            f"right_branch={row['right_main_branch_id']} "
            f"confidence={float(row['left_confidence']):.3f} "
            f"unknown_nodes={row['unknown_nodes']} "
            f"unknown_branches={row['unknown_branches']} "
            f"review={row['needs_manual_review']}"
        )
    print(f"wrote {output_csv}")
    print(f"wrote {len(rows)} side_assignment.json files and QC images under {qc_dir}")
    print("No graph node, branch, connectivity, or residual cycle was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
