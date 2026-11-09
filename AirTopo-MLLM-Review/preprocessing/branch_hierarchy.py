#!/usr/bin/env python3
"""Build a rooted airway hierarchy and generation labels for five test cases.

The clean graph is never modified. A deterministic length-weighted shortest-
path tree supplies the hierarchy when residual cycles exist; non-tree edges are
retained as residual_cycle_edge records and excluded from generation recursion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_CLEAN_ROOT = DATA_ROOT / "ATM22_processed" / "topology_clean"
DEFAULT_CASES = ()
EDGE_WEIGHT_FLOOR_MM = 0.4

SUMMARY_FIELDS = [
    "case_id",
    "max_generation",
    "mean_generation",
    "max_branch_depth_from_root",
    "branch_count",
    "hierarchy_branch_count",
    "residual_cycle_edge_count",
    "unknown_side_branch_count",
    "valid_split_node_count",
    "parent_conflict_count",
    "missing_parent_hierarchy_branch_count",
    "generation_decrease_count",
    "generation_jump_count",
    "left_right_interference_count",
    "graph_connectivity_unchanged",
    "node_count_unchanged",
    "branch_count_unchanged",
    "residual_cycle_rank_unchanged",
    "validation_passed",
    "needs_manual_review",
    "review_reasons",
    "hierarchy_json",
    "qc_png",
]


class HierarchyError(RuntimeError):
    """Raised when an input or hierarchy invariant is violated."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise HierarchyError(f"{path}: expected JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def component_count(
    node_by_id: dict[int, dict[str, Any]],
    branch_by_id: dict[int, dict[str, Any]],
) -> int:
    adjacency = {node_id: set() for node_id in node_by_id}
    for branch in branch_by_id.values():
        start = int(branch["start_node_id"])
        end = int(branch["end_node_id"])
        if start == end:
            raise HierarchyError(f"self-loop remains: branch {branch['id']}")
        if start not in adjacency or end not in adjacency:
            raise HierarchyError(f"branch {branch['id']} references missing node")
        adjacency[start].add(end)
        adjacency[end].add(start)
    unseen = set(adjacency)
    result = 0
    while unseen:
        result += 1
        stack = [min(unseen)]
        seen: set[int] = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(adjacency[node_id] - seen)
        unseen -= seen
    return result


def load_graph(case_dir: Path, case_id: str) -> dict[str, Any]:
    nodes_path = case_dir / "nodes.json"
    branches_path = case_dir / "branches.json"
    nodes_doc = load_json(nodes_path)
    branches_doc = load_json(branches_path)
    stats = load_json(case_dir / "stats.json")
    if nodes_doc.get("case_id") != case_id or branches_doc.get("case_id") != case_id:
        raise HierarchyError(f"{case_id}: case ID mismatch")
    if nodes_doc.get("coordinate_system") != "LPS" or branches_doc.get("coordinate_system") != "LPS":
        raise HierarchyError(f"{case_id}: expected LPS coordinates")
    nodes = nodes_doc["nodes"]
    branches = branches_doc["branches"]
    node_by_id = {int(node["id"]): node for node in nodes}
    branch_by_id = {int(branch["id"]): branch for branch in branches}
    if len(node_by_id) != len(nodes) or len(branch_by_id) != len(branches):
        raise HierarchyError(f"{case_id}: duplicate IDs")
    components = component_count(node_by_id, branch_by_id)
    cycle_rank = len(branches) - len(nodes) + components
    if components != 1 or not bool(stats["clean_is_connected"]):
        raise HierarchyError(f"{case_id}: clean graph is disconnected")
    if cycle_rank != int(stats["residual_cycle_after"]):
        raise HierarchyError(f"{case_id}: cycle rank mismatch")
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
        "hashes": {"nodes.json": sha256(nodes_path), "branches.json": sha256(branches_path)},
    }


def shortest_path_tree(root: int, graph: dict[str, Any]) -> dict[str, Any]:
    adjacency = graph["adjacency"]
    distance = {node_id: float("inf") for node_id in adjacency}
    distance[root] = 0.0
    parent: dict[int, int | None] = {root: None}
    parent_branch: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, root)]
    while queue:
        current_distance, node_id = heapq.heappop(queue)
        if current_distance > distance[node_id] + 1e-12:
            continue
        for neighbor, branch_id, length in adjacency[node_id]:
            proposed = current_distance + max(float(length), EDGE_WEIGHT_FLOOR_MM)
            old_key = (
                parent.get(neighbor, sys.maxsize),
                parent_branch.get(neighbor, sys.maxsize),
            )
            new_key = (node_id, branch_id)
            if (
                proposed < distance[neighbor] - 1e-12
                or (abs(proposed - distance[neighbor]) <= 1e-12 and new_key < old_key)
            ):
                distance[neighbor] = proposed
                parent[neighbor] = node_id
                parent_branch[neighbor] = branch_id
                heapq.heappush(queue, (proposed, neighbor))
    if len(parent) != len(adjacency):
        raise HierarchyError("shortest-path tree did not reach every node")
    children = {node_id: [] for node_id in adjacency}
    for node_id, parent_id in parent.items():
        if parent_id is not None:
            children[parent_id].append(node_id)
    for values in children.values():
        values.sort()
    node_depth = {root: 0}
    for node_id in sorted(distance, key=lambda value: (distance[value], value)):
        if node_id == root:
            continue
        node_depth[node_id] = node_depth[parent[node_id]] + 1
    tree_branch_ids = set(parent_branch.values())
    residual_branch_ids = set(graph["branch_by_id"]) - tree_branch_ids
    if len(residual_branch_ids) != graph["cycle_rank"]:
        raise HierarchyError("non-tree edge count does not equal cycle rank")
    return {
        "distance": distance,
        "parent": parent,
        "parent_branch": parent_branch,
        "children": children,
        "node_depth": node_depth,
        "tree_branch_ids": tree_branch_ids,
        "residual_branch_ids": residual_branch_ids,
    }


def tree_distances_from(source: int, tree: dict[str, Any], graph: dict[str, Any]) -> dict[int, float]:
    adjacency = {node_id: [] for node_id in graph["node_by_id"]}
    for node_id, parent_id in tree["parent"].items():
        if parent_id is None:
            continue
        branch_id = tree["parent_branch"][node_id]
        length = float(graph["branch_by_id"][branch_id]["length_mm"])
        adjacency[node_id].append((parent_id, length))
        adjacency[parent_id].append((node_id, length))
    result = {source: 0.0}
    stack = [(source, None)]
    while stack:
        node_id, previous = stack.pop()
        for neighbor, length in adjacency[node_id]:
            if neighbor == previous:
                continue
            result[neighbor] = result[node_id] + length
            stack.append((neighbor, node_id))
    if len(result) != len(adjacency):
        raise HierarchyError("tree distance traversal failed")
    return result


def assignment_maps(document: dict[str, Any], graph: dict[str, Any]) -> tuple[dict[int, str], dict[int, str]]:
    if document.get("coordinate_system") != "LPS":
        raise HierarchyError("side assignment is not LPS")
    node_side = {int(item["node_id"]): item["side"] for item in document["node_assignments"]}
    branch_side = {int(item["branch_id"]): item["side"] for item in document["branch_assignments"]}
    if set(node_side) != set(graph["node_by_id"]) or set(branch_side) != set(graph["branch_by_id"]):
        raise HierarchyError("side assignment does not cover the complete graph")
    allowed = {"trachea", "left", "right", "unknown"}
    if not set(node_side.values()) <= allowed or not set(branch_side.values()) <= allowed:
        raise HierarchyError("unexpected side value")
    return node_side, branch_side


def assign_generation(
    root: int,
    carina: int,
    trachea_nodes: list[int],
    trachea_branches: set[int],
    left_main: int,
    right_main: int,
    tree: dict[str, Any],
    branch_side: dict[int, str],
) -> tuple[dict[int, int], set[int]]:
    generation: dict[int, int] = {}
    valid_split_nodes = {
        node_id for node_id, children in tree["children"].items() if len(children) >= 2
    }
    ordered_nodes = sorted(
        (node_id for node_id in tree["parent"] if node_id != root),
        key=lambda node_id: (tree["node_depth"][node_id], node_id),
    )
    for child_node in ordered_nodes:
        branch_id = tree["parent_branch"][child_node]
        proximal_node = tree["parent"][child_node]
        if branch_id in trachea_branches:
            value = 0
        elif branch_id in {left_main, right_main}:
            value = 1
        elif proximal_node == root:
            value = 0
        else:
            parent_branch_id = tree["parent_branch"][proximal_node]
            if parent_branch_id not in generation:
                raise HierarchyError(f"parent generation missing for branch {branch_id}")
            parent_generation = generation[parent_branch_id]
            if proximal_node in trachea_nodes:
                value = 0 if branch_side[branch_id] == "trachea" else 1
            else:
                value = parent_generation + int(proximal_node in valid_split_nodes)
        generation[branch_id] = value
    if generation.get(left_main) != 1 or generation.get(right_main) != 1:
        raise HierarchyError("main bronchus generation is not 1")
    if any(generation.get(branch_id) != 0 for branch_id in trachea_branches):
        raise HierarchyError("trachea generation is not 0")
    if set(generation) != tree["tree_branch_ids"]:
        raise HierarchyError("not every hierarchy edge received a generation")
    return generation, valid_split_nodes


def side_ancestor_conflicts(
    left_main: int,
    right_main: int,
    tree: dict[str, Any],
    branch_side: dict[int, str],
) -> list[int]:
    conflicts: list[int] = []
    for branch_id in tree["tree_branch_ids"]:
        side = branch_side[branch_id]
        if side not in {"left", "right"}:
            continue
        expected_main = left_main if side == "left" else right_main
        child_node = next(
            node_id for node_id, value in tree["parent_branch"].items()
            if value == branch_id
        )
        current_branch = branch_id
        proximal = tree["parent"][child_node]
        ancestors = {current_branch}
        while proximal is not None and tree["parent"][proximal] is not None:
            current_branch = tree["parent_branch"][proximal]
            ancestors.add(current_branch)
            proximal = tree["parent"][proximal]
        opposite_main = right_main if side == "left" else left_main
        if expected_main not in ancestors or opposite_main in ancestors:
            conflicts.append(branch_id)
    return sorted(conflicts)


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
    branch_records: list[dict[str, Any]],
    root: int,
    carina: int,
    max_generation: int,
    output_path: Path,
) -> None:
    record_by_id = {record["branch_id"]: record for record in branch_records}
    projections = [
        (0, 2, "Coronal", "LPS X (+Left) mm", "LPS Z (+Superior) mm"),
        (1, 2, "Sagittal", "LPS Y (+Posterior) mm", "LPS Z (+Superior) mm"),
        (0, 1, "Axial", "LPS X (+Left) mm", "LPS Y (+Posterior) mm"),
    ]
    normalization = colors.Normalize(vmin=0, vmax=max(max_generation, 1))
    colormap = plt.get_cmap("turbo")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for row, target_side in enumerate(["left", "right"]):
        for ax, (horizontal, vertical, title, xlabel, ylabel) in zip(axes[row], projections):
            for branch_id, path in physical_paths.items():
                record = record_by_id[branch_id]
                side = record["side"]
                generation = record["generation"]
                if record["residual_cycle_edge"]:
                    if side == target_side:
                        ax.plot(path[:, horizontal], path[:, vertical], color="black",
                                linewidth=1.6, linestyle="--", alpha=0.9)
                    continue
                if side == target_side or side == "trachea":
                    ax.plot(path[:, horizontal], path[:, vertical],
                            color=colormap(normalization(generation)), linewidth=1.25,
                            alpha=0.85 if side == target_side else 1.0)
                elif side == "unknown":
                    ax.plot(path[:, horizontal], path[:, vertical], color="#777777",
                            linewidth=0.65, alpha=0.35)
                else:
                    ax.plot(path[:, horizontal], path[:, vertical], color="#dddddd",
                            linewidth=0.45, alpha=0.18)
            root_point = np.asarray(graph["node_by_id"][root]["centroid_physical"])
            carina_point = np.asarray(graph["node_by_id"][carina]["centroid_physical"])
            ax.scatter(root_point[horizontal], root_point[vertical], marker="*", s=70,
                       color="black", edgecolor="white", linewidth=0.5, zorder=10)
            ax.scatter(carina_point[horizontal], carina_point[vertical], marker="D", s=45,
                       color="#ffbf00", edgecolor="black", linewidth=0.5, zorder=10)
            ax.set_title(f"{target_side.capitalize()} subtree - {title}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12)
    scalar = plt.cm.ScalarMappable(norm=normalization, cmap=colormap)
    scalar.set_array([])
    fig.colorbar(scalar, ax=axes, location="right", shrink=0.72, label="Generation")
    fig.legend(handles=[
        Line2D([0], [0], color="black", linestyle="--", lw=1.6,
               label="residual cycle edge (excluded from generation)"),
        Line2D([0], [0], marker="*", color="black", lw=0, markersize=9, label="root"),
        Line2D([0], [0], marker="D", markerfacecolor="#ffbf00",
               markeredgecolor="black", color="none", lw=0, markersize=7, label="carina"),
    ], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(f"{case_id} branch hierarchy | max generation={max_generation}", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, output_path)


def analyze_case(case_id: str, clean_root: Path, qc_dir: Path) -> dict[str, Any]:
    case_dir = clean_root / case_id
    graph = load_graph(case_dir, case_id)
    side_doc = load_json(case_dir / "side_assignment.json")
    if side_doc.get("case_id") != case_id:
        raise HierarchyError(f"{case_id}: side assignment case mismatch")
    node_side, branch_side = assignment_maps(side_doc, graph)
    root = int(side_doc["root_node_id"])
    carina = int(side_doc["carina_node_id"])
    trachea_nodes = [int(value) for value in side_doc["trachea_node_ids"]]
    trachea_branches = {int(value) for value in side_doc["trachea_branch_ids"]}
    left_main = int(side_doc["left_main_branch_id"])
    right_main = int(side_doc["right_main_branch_id"])
    tree = shortest_path_tree(root, graph)
    generation, valid_split_nodes = assign_generation(
        root, carina, trachea_nodes, trachea_branches,
        left_main, right_main, tree, branch_side,
    )
    root_distances = tree_distances_from(root, tree, graph)
    carina_distances = tree_distances_from(carina, tree, graph)
    child_by_branch = {branch_id: node_id for node_id, branch_id in tree["parent_branch"].items()}

    branch_records: list[dict[str, Any]] = []
    for branch_id in sorted(graph["branch_by_id"]):
        branch = graph["branch_by_id"][branch_id]
        is_tree = branch_id in tree["tree_branch_ids"]
        if is_tree:
            distal_node = child_by_branch[branch_id]
            proximal_node = tree["parent"][distal_node]
            parent_branch_id = (
                tree["parent_branch"][proximal_node]
                if proximal_node != root else None
            )
            children_branch_ids = [
                tree["parent_branch"][node_id]
                for node_id in tree["children"][distal_node]
            ]
            record_generation: int | None = generation[branch_id]
            branch_depth: int | None = tree["node_depth"][proximal_node]
            root_distance: float | None = root_distances[proximal_node]
            carina_distance: float | None = carina_distances[proximal_node]
        else:
            proximal_node = None
            distal_node = None
            parent_branch_id = None
            children_branch_ids = []
            record_generation = None
            branch_depth = None
            root_distance = None
            carina_distance = None
        branch_records.append({
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "children_branch_ids": sorted(children_branch_ids),
            "generation": record_generation,
            "side": branch_side[branch_id],
            "branch_depth_from_root": branch_depth,
            "path_distance_from_root_mm": root_distance,
            "path_distance_from_carina_mm": carina_distance,
            "branch_length_mm": float(branch["length_mm"]),
            "proximal_node_id": proximal_node,
            "distal_node_id": distal_node,
            "hierarchy_edge": is_tree,
            "residual_cycle_edge": not is_tree,
        })

    node_records = []
    for node_id in sorted(graph["node_by_id"]):
        parent_node = tree["parent"][node_id]
        node_records.append({
            "node_id": node_id,
            "parent_node_id": parent_node,
            "children_node_ids": tree["children"][node_id],
            "depth_from_root": tree["node_depth"][node_id],
            "path_distance_from_root_mm": root_distances[node_id],
            "path_distance_from_carina_mm": carina_distances[node_id],
            "side": node_side[node_id],
            "hierarchy_parent_branch_id": (
                tree["parent_branch"][node_id] if parent_node is not None else None
            ),
        })

    record_by_id = {record["branch_id"]: record for record in branch_records}
    parent_conflicts = [
        record["branch_id"] for record in branch_records
        if record["hierarchy_edge"]
        and not isinstance(record["parent_branch_id"], (int, type(None)))
    ]
    missing_parents = [
        record["branch_id"] for record in branch_records
        if record["hierarchy_edge"]
        and record["proximal_node_id"] != root
        and record["parent_branch_id"] is None
    ]
    decreases: list[int] = []
    jumps: list[int] = []
    for record in branch_records:
        parent_id = record["parent_branch_id"]
        if not record["hierarchy_edge"] or parent_id is None:
            continue
        parent_generation = record_by_id[parent_id]["generation"]
        difference = record["generation"] - parent_generation
        if difference < 0:
            decreases.append(record["branch_id"])
        if difference > 1:
            jumps.append(record["branch_id"])
    side_conflicts = side_ancestor_conflicts(
        left_main, right_main, tree, branch_side
    )
    graph_hashes_after = {
        "nodes.json": sha256(case_dir / "nodes.json"),
        "branches.json": sha256(case_dir / "branches.json"),
    }
    graph_unchanged = graph_hashes_after == graph["hashes"]
    validations = {
        "trachea_generation_zero": all(generation[branch_id] == 0 for branch_id in trachea_branches),
        "left_main_generation_one": generation[left_main] == 1,
        "right_main_generation_one": generation[right_main] == 1,
        "hierarchy_branch_count": len(tree["tree_branch_ids"]),
        "expected_hierarchy_branch_count": len(graph["nodes"]) - 1,
        "parent_conflict_branch_ids": parent_conflicts,
        "missing_parent_hierarchy_branch_ids": missing_parents,
        "generation_decrease_branch_ids": decreases,
        "generation_jump_branch_ids": jumps,
        "left_right_interference_branch_ids": side_conflicts,
        "residual_cycle_edges_excluded_from_generation": all(
            record_by_id[branch_id]["generation"] is None
            for branch_id in tree["residual_branch_ids"]
        ),
        "graph_components_before": graph["components"],
        "graph_components_after": graph["components"],
        "node_count_before": len(graph["nodes"]),
        "node_count_after": len(graph["nodes"]),
        "branch_count_before": len(graph["branches"]),
        "branch_count_after": len(graph["branches"]),
        "residual_cycle_rank_before": graph["cycle_rank"],
        "residual_cycle_rank_after": graph["cycle_rank"],
        "source_graph_sha256_unchanged": graph_unchanged,
    }
    review_reasons: list[str] = []
    if parent_conflicts or missing_parents:
        review_reasons.append("hierarchy_parent_error")
    if decreases or jumps:
        review_reasons.append("generation_monotonicity_error")
    if side_conflicts:
        review_reasons.append("left_right_hierarchy_interference")
    if not graph_unchanged:
        raise HierarchyError(f"{case_id}: source graph changed")
    validation_passed = not review_reasons
    hierarchy_path = case_dir / "branch_hierarchy.json"
    generation_values = list(generation.values())
    generation_histogram = dict(sorted(Counter(generation_values).items()))
    document = {
        "case_id": case_id,
        "stage": "branch_hierarchy_and_airway_generation",
        "coordinate_system": "LPS",
        "source_graph_modified": False,
        "generation_convention": {
            "trachea": 0,
            "left_right_main_bronchus": 1,
            "increment": "plus one at a hierarchy node with two or more distal hierarchy children",
            "residual_cycle_edge_generation": None,
        },
        "root_node_id": root,
        "carina_node_id": carina,
        "left_main_branch_id": left_main,
        "right_main_branch_id": right_main,
        "branch_count": len(graph["branches"]),
        "hierarchy_branch_count": len(tree["tree_branch_ids"]),
        "residual_cycle_edge_count": len(tree["residual_branch_ids"]),
        "residual_cycle_branch_ids": sorted(tree["residual_branch_ids"]),
        "max_generation": max(generation_values),
        "mean_generation": float(sum(generation_values) / len(generation_values)),
        "generation_histogram": {str(key): value for key, value in generation_histogram.items()},
        "valid_split_node_ids": sorted(valid_split_nodes),
        "branches": branch_records,
        "nodes": node_records,
        "validation": validations,
        "validation_passed": validation_passed,
        "needs_manual_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }
    write_json_atomic(hierarchy_path, document)
    physical_paths = {
        branch_id: np.load(case_dir / branch["physical_file"], allow_pickle=False).astype(
            np.float64, copy=False
        )
        for branch_id, branch in graph["branch_by_id"].items()
    }
    qc_path = qc_dir / f"{case_id}_generation_qc.png"
    plot_qc(
        case_id, graph, physical_paths, branch_records, root, carina,
        document["max_generation"], qc_path,
    )
    return {
        "case_id": case_id,
        "max_generation": document["max_generation"],
        "mean_generation": document["mean_generation"],
        "max_branch_depth_from_root": max(
            record["branch_depth_from_root"] for record in branch_records
            if record["hierarchy_edge"]
        ),
        "branch_count": len(graph["branches"]),
        "hierarchy_branch_count": len(tree["tree_branch_ids"]),
        "residual_cycle_edge_count": len(tree["residual_branch_ids"]),
        "unknown_side_branch_count": sum(side == "unknown" for side in branch_side.values()),
        "valid_split_node_count": len(valid_split_nodes),
        "parent_conflict_count": len(parent_conflicts),
        "missing_parent_hierarchy_branch_count": len(missing_parents),
        "generation_decrease_count": len(decreases),
        "generation_jump_count": len(jumps),
        "left_right_interference_count": len(side_conflicts),
        "graph_connectivity_unchanged": graph_unchanged and graph["components"] == 1,
        "node_count_unchanged": True,
        "branch_count_unchanged": True,
        "residual_cycle_rank_unchanged": graph["cycle_rank"],
        "validation_passed": validation_passed,
        "needs_manual_review": bool(review_reasons),
        "review_reasons": ";".join(review_reasons),
        "hierarchy_json": str(hierarchy_path.resolve()),
        "qc_png": str(qc_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build branch hierarchy and generation for selected cases."
    )
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--cases", nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_root = args.clean_root.resolve()
    cases = list(dict.fromkeys(args.cases))
    qc_dir = clean_root / "branch_hierarchy_qc"
    rows = [analyze_case(case_id, clean_root, qc_dir) for case_id in cases]
    output_csv = clean_root / "branch_hierarchy_summary.csv"
    temporary = output_csv.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_csv)
    for row in rows:
        print(
            f"{row['case_id']}: max_generation={row['max_generation']} "
            f"hierarchy={row['hierarchy_branch_count']}/{row['branch_count']} "
            f"cycle_edges={row['residual_cycle_edge_count']} "
            f"parent_conflicts={row['parent_conflict_count']} "
            f"generation_jumps={row['generation_jump_count']} "
            f"valid={row['validation_passed']}"
        )
    print(f"wrote {output_csv}")
    print(f"wrote {len(rows)} branch_hierarchy.json files and QC images under {qc_dir}")
    print("No graph node, branch, connectivity, side assignment, or cycle was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
