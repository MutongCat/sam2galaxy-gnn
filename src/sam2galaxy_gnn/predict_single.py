from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .artifacts import ModelArtifact
from .graph_builder import GraphInput, load_selected_targets, select_prediction_targets
from .runtime import TARGET_NAMES, build_inference_batches, build_model_from_config, load_scaler_stats, run_model_on_batches


def _to_rows(payload: dict[str, Any], *, sam_id: int | None = None) -> list[dict[str, Any]]:
    pred = payload["prediction"]
    sigma = payload["prediction_sigma"]
    rows: list[dict[str, Any]] = []
    for i in range(pred.shape[0]):
        row = {
            "node_index": int(payload["node_index"][i]),
            "tree_id": int(payload["tree_id"][i]),
            "z_index": int(payload["z_index"][i]),
            "redshift": float(payload["redshift"][i]),
            "quench_probability": float(payload["quench_probability"][i]),
            "gas_floor_probability": float(payload["gas_floor_probability"][i]),
            "predicted_quenched": int(float(payload["quench_probability"][i]) >= 0.5),
            "predicted_gas_floor": int(float(payload["gas_floor_probability"][i]) >= 0.5),
        }
        if sam_id is not None:
            row["sam_id"] = int(sam_id)
        for j, name in enumerate(TARGET_NAMES):
            row[name] = float(pred[i, j])
            row[f"{name}_sigma"] = float(sigma[i, j])
        rows.append(row)
    return rows


def predict_single(
    graph_input: GraphInput,
    artifact: ModelArtifact,
    *,
    selected_targets_path: str | Path | None = None,
    selected_target_sam_id: int | None = None,
) -> dict[str, Any]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = graph_input.payload
    if not hasattr(graph, "u"):
        raise RuntimeError("Graph input is missing SAM parameters 'u'. Provide --sam-parameters or embed graph.u.")
    targets = (
        load_selected_targets(graph_input, selected_targets_path, sam_id=selected_target_sam_id)
        if selected_targets_path is not None
        else select_prediction_targets(graph_input)
    )
    model = build_model_from_config(artifact.config_path, artifact.checkpoint_path, device=device)
    scaler = load_scaler_stats(artifact.scaler_path, device=device)
    batches = build_inference_batches(
        graph,
        y_indices=targets.y_indices,
        z_indices=targets.z_indices,
        matched_z_values=targets.matched_z_values,
    )
    payload = run_model_on_batches(model, batches, scaler=scaler, device=device)
    rows = _to_rows(payload, sam_id=selected_target_sam_id)
    return {
        "mode": "single",
        "artifact_name": artifact.name,
        "input_format": graph_input.source_format,
        "input_path": str(graph_input.source_path),
        "num_rows": len(rows),
        "rows": rows,
    }


def write_single_prediction_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return output
