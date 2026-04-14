from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math


TARGET_NAMES = [
    "stellar_mass",
    "sdss_z_band_luminosity",
    "angular_momentum",
    "gas_metal_mass",
    "specific_star_formation_rate",
]

DEFAULT_SSFR_QUENCH_THRESHOLD = -2.0
DEFAULT_ZERO_FLOOR_GAS = -5.0


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_summary_json(summary: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return output


def _key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("sam_id", -1)),
        int(row["node_index"]),
        int(row["z_index"]),
    )


def align_prediction_rows(
    *,
    prediction_rows: list[dict[str, Any]],
    selected_target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted = {_key(row) for row in selected_target_rows}
    aligned = [row for row in prediction_rows if _key(row) in wanted]
    aligned.sort(key=lambda row: (int(row.get("sam_id", -1)), int(row["z_index"]), int(row["tree_id"]), int(row["node_index"])))
    return aligned


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _variance(values: list[float], mean: float) -> float:
    if not values:
        return float("nan")
    return sum((v - mean) ** 2 for v in values) / len(values)


def _f1_score(truth: list[int], pred: list[int]) -> float:
    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def build_joined_eval(
    *,
    prediction_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    truth_by_key = {_key(row): row for row in truth_rows}
    joined: list[dict[str, Any]] = []
    for pred in prediction_rows:
        key = _key(pred)
        if key not in truth_by_key:
            continue
        truth = truth_by_key[key]
        row = dict(pred)
        for name in TARGET_NAMES:
            row[f"{name}_truth"] = float(truth[name])
            row[f"{name}_error"] = float(pred[name]) - float(truth[name])
        row["quench_truth"] = int(truth["quench_truth"])
        row["gas_floor_truth"] = int(truth["gas_floor_truth"])
        joined.append(row)

    metrics: dict[str, Any] = {
        "num_prediction_rows": len(prediction_rows),
        "num_truth_rows": len(truth_rows),
        "num_joined_rows": len(joined),
        "coverage_fraction": float(len(joined)) / float(len(truth_rows)) if truth_rows else float("nan"),
        "sam_ids": sorted({int(row["sam_id"]) for row in truth_rows}),
        "targets": {},
        "classification": {},
    }

    def _regression_metrics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
        pred_vals = [float(row[name]) for row in rows]
        truth_vals = [float(row[f"{name}_truth"]) for row in rows]
        errors = [p - t for p, t in zip(pred_vals, truth_vals)]
        truth_mean = _mean(truth_vals)
        sst = sum((t - truth_mean) ** 2 for t in truth_vals)
        sse = sum(e**2 for e in errors)
        return {
            "regression_row_count": len(rows),
            "bias": _mean(errors),
            "sigma": math.sqrt(_variance(errors, _mean(errors))),
            "mae": _mean([abs(e) for e in errors]),
            "r2": float("nan") if sst == 0 else 1.0 - sse / sst,
        }

    for name in TARGET_NAMES:
        regression_rows = joined
        if name == "gas_metal_mass":
            regression_rows = [
                row
                for row in joined
                if int(row["gas_floor_truth"]) == 0 and float(row[name]) > (DEFAULT_ZERO_FLOOR_GAS + 1e-8)
            ]
        elif name == "specific_star_formation_rate":
            regression_rows = [
                row
                for row in joined
                if int(row["quench_truth"]) == 0 and float(row[name]) > (DEFAULT_SSFR_QUENCH_THRESHOLD + 1e-8)
            ]
        metrics["targets"][name] = _regression_metrics(regression_rows, name)

    q_truth = [int(row["quench_truth"]) for row in joined]
    q_pred = [int(row["predicted_quenched"]) for row in joined]
    g_truth = [int(row["gas_floor_truth"]) for row in joined]
    g_pred = [int(row["predicted_gas_floor"]) for row in joined]
    metrics["classification"]["quench_f1"] = _f1_score(q_truth, q_pred)
    metrics["classification"]["gas_floor_f1"] = _f1_score(g_truth, g_pred)
    return joined, metrics
