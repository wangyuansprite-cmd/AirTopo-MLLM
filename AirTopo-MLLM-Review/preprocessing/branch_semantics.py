#!/usr/bin/env python3
"""Build generation-based branch-level semantic labels from frozen Stage-5 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_CLEAN_ROOT = DATA_ROOT / "ATM22_processed" / "topology_clean"
CASE_PATTERN = re.compile(r"^ATM_\d+$")
QC_CASES = ()
LABELS = [
    "trachea", "left_central", "right_central", "left_peripheral",
    "right_peripheral", "unknown_central", "unknown_peripheral", "residual_cycle",
]
SUMMARY_FIELDS = [
    "case_id", "total_hierarchy_branches", "trachea_branches",
    "left_central_branches", "right_central_branches",
    "left_peripheral_branches", "right_peripheral_branches",
    "unknown_central_branches", "unknown_peripheral_branches",
    "residual_cycle_edges", "central_total_branches", "peripheral_total_branches",
]
GLOBAL_FIELDS = [
    "semantic_label", "total_branch_count", "case_count_present",
    "case_coverage_percent", "branch_percent",
]


class SemanticError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def valid_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and np.isfinite(value) and value.is_integer()


def semantic_for(branch: dict[str, Any]) -> tuple[str | None, str]:
    generation = branch.get("generation")
    side = str(branch.get("side", "unknown"))
    if bool(branch.get("residual_cycle_edge")) or generation is None:
        return None, "residual_cycle"
    if not bool(branch.get("hierarchy_edge")):
        raise SemanticError(f"branch {branch.get('branch_id')}: non-residual non-hierarchy edge")
    if not valid_integer(generation) or int(generation) < 0:
        raise SemanticError(f"branch {branch.get('branch_id')}: invalid generation {generation!r}")
    generation = int(generation)
    region = "central" if generation <= 6 else "peripheral"
    if generation == 0:
        if side != "trachea":
            raise SemanticError(f"branch {branch.get('branch_id')}: G0 side is {side}, expected trachea")
        return region, "trachea"
    if side == "left":
        return region, f"left_{region}"
    if side == "right":
        return region, f"right_{region}"
    if side == "unknown":
        return region, f"unknown_{region}"
    if side == "trachea":
        raise SemanticError(f"branch {branch.get('branch_id')}: trachea side has G{generation}")
    raise SemanticError(f"branch {branch.get('branch_id')}: unsupported side {side!r}")


def plot_qc(case_dir: Path, case_id: str, semantic_records: list[dict[str, Any]]) -> None:
    with (case_dir / "branches.json").open("r", encoding="utf-8") as handle:
        source_branches = json.load(handle)["branches"]
    source_by_id = {int(branch["id"]): branch for branch in source_branches}
    record_by_id = {int(record["branch_id"]): record for record in semantic_records}
    if set(source_by_id) != set(record_by_id):
        raise SemanticError(f"{case_id}: QC branch IDs differ from semantic records")
    colors = {
        "trachea": "#0d47a1", "left_central": "#b71c1c", "right_central": "#1b5e20",
        "left_peripheral": "#ef5350", "right_peripheral": "#66bb6a",
        "unknown_central": "#ff8f00", "unknown_peripheral": "#ffca28",
        "residual_cycle": "#616161",
    }
    widths = {
        "trachea": 3.4, "left_central": 2.4, "right_central": 2.4,
        "left_peripheral": 1.0, "right_peripheral": 1.0,
        "unknown_central": 2.2, "unknown_peripheral": 1.6, "residual_cycle": 2.6,
    }
    order = [
        "left_peripheral", "right_peripheral", "left_central", "right_central",
        "unknown_peripheral", "unknown_central", "trachea", "residual_cycle",
    ]
    physical = {
        branch_id: np.load(case_dir / branch["physical_file"], allow_pickle=False).astype(np.float64, copy=False)
        for branch_id, branch in source_by_id.items()
    }
    projections = [
        (0, 2, "Coronal", "LPS X (+Left) mm", "LPS Z (+Superior) mm"),
        (1, 2, "Sagittal", "LPS Y (+Posterior) mm", "LPS Z (+Superior) mm"),
        (0, 1, "Axial", "LPS X (+Left) mm", "LPS Y (+Posterior) mm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for ax, (horizontal, vertical, title, xlabel, ylabel) in zip(axes, projections):
        for label in order:
            for branch_id, record in record_by_id.items():
                if record["semantic_label"] != label:
                    continue
                path = physical[branch_id]
                ax.plot(
                    path[:, horizontal], path[:, vertical], color=colors[label],
                    lw=widths[label], alpha=0.95 if label != "residual_cycle" else 0.85,
                    ls="--" if label == "residual_cycle" else "-",
                )
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.15)
    counts = Counter(record["semantic_label"] for record in semantic_records)
    handles = [
        Line2D([0], [0], color=colors[label], lw=widths[label],
               ls="--" if label == "residual_cycle" else "-",
               label=f"{label.replace('_', ' ')} ({counts[label]})")
        for label in LABELS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(
        f"{case_id} - branch-level semantics (central G0-G6; peripheral G7+)",
        fontsize=13, y=1.01,
    )
    output = case_dir / "semantic_branch_qc.png"
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=190, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    args = parser.parse_args()
    clean_root = args.clean_root.resolve()
    case_dirs = sorted(
        path for path in clean_root.iterdir()
        if path.is_dir() and CASE_PATTERN.fullmatch(path.name)
        and (path / "branch_hierarchy.json").is_file()
    )
    if len(case_dirs) != 297:
        raise SemanticError(f"expected 297 Stage-5 cases, found {len(case_dirs)}")
    hierarchy_paths = [case_dir / "branch_hierarchy.json" for case_dir in case_dirs]
    hashes_before = {case_dir.name: sha256(path) for case_dir, path in zip(case_dirs, hierarchy_paths)}

    semantic_documents: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    global_counts: Counter[str] = Counter()
    label_cases: dict[str, set[str]] = defaultdict(set)
    validation: dict[str, Any] = {
        "expected_case_count": 297, "processed_case_count": len(case_dirs),
        "case_failures": [], "unlabeled_hierarchy_branches": [],
        "g0_not_trachea": [], "g1_g6_not_central": [], "g7plus_not_peripheral": [],
        "left_labeled_right": [], "right_labeled_left": [],
        "residual_not_residual_cycle": [], "null_generation_assigned_region": [],
        "semantic_branch_count_mismatch": [], "hierarchy_branch_count_mismatch": [],
        "unknown_semantic_labels": [],
    }
    input_total_branches = 0
    input_total_hierarchy = 0
    output_total_branches = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        try:
            with (case_dir / "branch_hierarchy.json").open("r", encoding="utf-8") as handle:
                hierarchy = json.load(handle)
            if hierarchy.get("case_id") != case_id or hierarchy.get("coordinate_system") != "LPS":
                raise SemanticError(f"{case_id}: invalid hierarchy identity/coordinate system")
            records: list[dict[str, Any]] = []
            case_counts: Counter[str] = Counter()
            hierarchy_count = 0
            for branch in hierarchy["branches"]:
                branch_id = int(branch["branch_id"])
                generation = branch.get("generation")
                side = str(branch.get("side", "unknown"))
                hierarchy_edge = bool(branch.get("hierarchy_edge"))
                residual_edge = bool(branch.get("residual_cycle_edge"))
                region, label = semantic_for(branch)
                record = {
                    "branch_id": branch_id,
                    "generation": None if generation is None else int(generation),
                    "side": side, "region": region, "semantic_label": label,
                    "hierarchy_edge": hierarchy_edge, "residual_cycle_edge": residual_edge,
                }
                records.append(record)
                case_counts[label] += 1; global_counts[label] += 1; label_cases[label].add(case_id)
                if hierarchy_edge:
                    hierarchy_count += 1
                    if not label:
                        validation["unlabeled_hierarchy_branches"].append({"case_id": case_id, "branch_id": branch_id})
                if generation == 0 and label != "trachea":
                    validation["g0_not_trachea"].append({"case_id": case_id, "branch_id": branch_id, "label": label})
                if generation is not None and 1 <= int(generation) <= 6 and region != "central":
                    validation["g1_g6_not_central"].append({"case_id": case_id, "branch_id": branch_id})
                if generation is not None and int(generation) >= 7 and region != "peripheral":
                    validation["g7plus_not_peripheral"].append({"case_id": case_id, "branch_id": branch_id})
                if side == "left" and label.startswith("right_"):
                    validation["left_labeled_right"].append({"case_id": case_id, "branch_id": branch_id})
                if side == "right" and label.startswith("left_"):
                    validation["right_labeled_left"].append({"case_id": case_id, "branch_id": branch_id})
                if residual_edge and label != "residual_cycle":
                    validation["residual_not_residual_cycle"].append({"case_id": case_id, "branch_id": branch_id})
                if generation is None and (region is not None or label != "residual_cycle"):
                    validation["null_generation_assigned_region"].append({"case_id": case_id, "branch_id": branch_id})
                if label not in LABELS:
                    validation["unknown_semantic_labels"].append({"case_id": case_id, "branch_id": branch_id, "label": label})
            input_total_branches += len(hierarchy["branches"])
            input_total_hierarchy += int(hierarchy["hierarchy_branch_count"])
            output_total_branches += len(records)
            if len(records) != len(hierarchy["branches"]):
                validation["semantic_branch_count_mismatch"].append(case_id)
            if hierarchy_count != int(hierarchy["hierarchy_branch_count"]):
                validation["hierarchy_branch_count_mismatch"].append({
                    "case_id": case_id, "document": int(hierarchy["hierarchy_branch_count"]),
                    "observed": hierarchy_count,
                })
            semantic_documents[case_id] = {
                "case_id": case_id,
                "semantic_definition": {
                    "central_generation_range": [0, 6],
                    "peripheral_generation_min": 7,
                    "definition_version": "generation_based_v1",
                    "definition_note": "generation-based operational research definition; not a unique medical standard",
                    "residual_cycle_policy": "region=null; excluded from normal semantic hierarchy",
                },
                "source_branch_hierarchy_sha256": hashes_before[case_id],
                "branches": records,
            }
            central_total = sum(case_counts[label] for label in [
                "trachea", "left_central", "right_central", "unknown_central",
            ])
            peripheral_total = sum(case_counts[label] for label in [
                "left_peripheral", "right_peripheral", "unknown_peripheral",
            ])
            summary_rows.append({
                "case_id": case_id,
                "total_hierarchy_branches": hierarchy_count,
                "trachea_branches": case_counts["trachea"],
                "left_central_branches": case_counts["left_central"],
                "right_central_branches": case_counts["right_central"],
                "left_peripheral_branches": case_counts["left_peripheral"],
                "right_peripheral_branches": case_counts["right_peripheral"],
                "unknown_central_branches": case_counts["unknown_central"],
                "unknown_peripheral_branches": case_counts["unknown_peripheral"],
                "residual_cycle_edges": case_counts["residual_cycle"],
                "central_total_branches": central_total,
                "peripheral_total_branches": peripheral_total,
            })
        except Exception as exc:
            validation["case_failures"].append({
                "case_id": case_id, "error_type": type(exc).__name__, "error_message": str(exc),
            })

    if validation["case_failures"]:
        validation["validation_passed"] = False
        write_json_atomic(clean_root / "semantic_branch_validation.json", validation)
        raise SemanticError(f"semantic construction failed for {len(validation['case_failures'])} cases")

    for case_dir in case_dirs:
        write_json_atomic(case_dir / "semantic_branches.json", semantic_documents[case_dir.name])
    for case_id in QC_CASES:
        plot_qc(clean_root / case_id, case_id, semantic_documents[case_id]["branches"])
    write_csv_atomic(clean_root / "semantic_branch_summary.csv", SUMMARY_FIELDS, summary_rows)
    global_rows = [
        {
            "semantic_label": label,
            "total_branch_count": global_counts[label],
            "case_count_present": len(label_cases[label]),
            "case_coverage_percent": 100.0 * len(label_cases[label]) / len(case_dirs),
            "branch_percent": 100.0 * global_counts[label] / output_total_branches,
        }
        for label in LABELS
    ]
    write_csv_atomic(clean_root / "semantic_branch_global_summary.csv", GLOBAL_FIELDS, global_rows)

    hashes_after = {case_dir.name: sha256(path) for case_dir, path in zip(case_dirs, hierarchy_paths)}
    changed = [case_id for case_id in hashes_before if hashes_before[case_id] != hashes_after[case_id]]
    check_keys = [
        "case_failures", "unlabeled_hierarchy_branches", "g0_not_trachea",
        "g1_g6_not_central", "g7plus_not_peripheral", "left_labeled_right",
        "right_labeled_left", "residual_not_residual_cycle",
        "null_generation_assigned_region", "semantic_branch_count_mismatch",
        "hierarchy_branch_count_mismatch", "unknown_semantic_labels",
    ]
    validation.update({
        "input_total_branch_records": input_total_branches,
        "output_total_semantic_branch_records": output_total_branches,
        "input_total_hierarchy_branches": input_total_hierarchy,
        "normal_semantic_hierarchy_branches": sum(global_counts[label] for label in LABELS if label != "residual_cycle"),
        "residual_cycle_edges": global_counts["residual_cycle"],
        "semantic_branch_total_matches_input": output_total_branches == input_total_branches,
        "normal_semantic_total_matches_hierarchy": (
            sum(global_counts[label] for label in LABELS if label != "residual_cycle") == input_total_hierarchy
        ),
        "branch_hierarchy_sha256_before": hashes_before,
        "branch_hierarchy_sha256_after": hashes_after,
        "branch_hierarchy_files_unchanged": not changed,
        "changed_branch_hierarchy_files": changed,
        "violation_counts": {key: len(validation[key]) for key in check_keys},
    })
    validation["validation_passed"] = (
        len(case_dirs) == 297
        and validation["semantic_branch_total_matches_input"]
        and validation["normal_semantic_total_matches_hierarchy"]
        and validation["branch_hierarchy_files_unchanged"]
        and not any(validation["violation_counts"].values())
    )
    write_json_atomic(clean_root / "semantic_branch_validation.json", validation)
    if not validation["validation_passed"]:
        raise SemanticError("semantic validation failed; inspect semantic_branch_validation.json")
    print(f"processed_cases={len(case_dirs)}/297")
    print(f"all_branch_records={output_total_branches} hierarchy={input_total_hierarchy} residual={global_counts['residual_cycle']}")
    for label in LABELS:
        print(f"{label}={global_counts[label]} cases={len(label_cases[label])}")
    print("validation_passed=true; all branch_hierarchy.json SHA-256 values unchanged")
    print(f"qc_cases={','.join(QC_CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
