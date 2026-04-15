from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .artifacts import ModelArtifact
from .graph_builder import GraphInput, load_selected_targets, select_prediction_targets
from .runtime import (
    TARGET_NAMES,
    build_inference_batches,
    build_model_from_config,
    load_scaler_stats,
    run_model_on_batches,
)


SSFR_QUENCH_THRESHOLD = -2.0
GAS_FLOOR_VALUE = -5.0


def _majority_mean_from_hard(
    point_preds: list["torch.Tensor"],
    hard_masks: list["torch.Tensor"],
    *,
    floor_value: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    pred_stack = torch.stack(point_preds, dim=0).to(torch.float32)
    mask_stack = torch.stack(hard_masks, dim=0).to(torch.bool)
    votes = mask_stack.sum(dim=0)
    majority_mask = votes > (pred_stack.shape[0] / 2.0)

    pred_out = torch.full_like(pred_stack[0], float(floor_value))
    cont_mask = ~mask_stack
    cont_counts = cont_mask.sum(dim=0)
    cont_sum = torch.where(cont_mask, pred_stack, torch.zeros_like(pred_stack)).sum(dim=0)
    cont_mean = torch.where(
        cont_counts > 0,
        cont_sum / cont_counts.clamp_min(1),
        torch.full_like(cont_sum, float(floor_value)),
    )
    pred_out[~majority_mask] = cont_mean[~majority_mask]
    prob_out = votes.to(torch.float32) / float(pred_stack.shape[0])
    return pred_out, prob_out


def _total_sigma(
    mean_stack: "torch.Tensor",
    sigma_stack: "torch.Tensor",
    *,
    final_mean: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    aleatoric = (sigma_stack.to(torch.float32) ** 2).mean(dim=0)
    epistemic = ((mean_stack.to(torch.float32) - final_mean.unsqueeze(0)) ** 2).mean(dim=0)
    return torch.sqrt((aleatoric + epistemic).clamp_min(0.0))


def predict_ensemble(
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
    if artifact.checkpoint_list_path is None:
        raise RuntimeError("Ensemble artifact is missing a checkpoint list")

    ckpts = [
        Path(line.strip())
        for line in artifact.checkpoint_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    targets = (
        load_selected_targets(graph_input, selected_targets_path, sam_id=selected_target_sam_id)
        if selected_targets_path is not None
        else select_prediction_targets(graph_input)
    )
    scaler = load_scaler_stats(artifact.scaler_path, device=device)
    batches = build_inference_batches(
        graph,
        y_indices=targets.y_indices,
        z_indices=targets.z_indices,
        matched_z_values=targets.matched_z_values,
    )
    predictions = []
    prediction_sigma = []
    quench_probs = []
    gas_probs = []
    ssfr_point_preds = []
    ssfr_point_sigma = []
    ssfr_hard_masks = []
    gas_point_preds = []
    gas_point_sigma = []
    gas_hard_masks = []
    for ckpt in ckpts:
        model = build_model_from_config(artifact.config_path, ckpt, device=device)
        payload = run_model_on_batches(model, batches, scaler=scaler, device=device)
        predictions.append(payload["prediction"])
        prediction_sigma.append(payload["prediction_sigma"])
        quench_probs.append(payload["quench_probability"])
        gas_probs.append(payload["gas_floor_probability"])
        ssfr_point_preds.append(payload["prediction"][:, 4])
        ssfr_point_sigma.append(payload["prediction_sigma"][:, 4])
        ssfr_hard_masks.append(payload["prediction"][:, 4] <= (SSFR_QUENCH_THRESHOLD + 1e-8))
        gas_point_preds.append(payload["prediction"][:, 3])
        gas_point_sigma.append(payload["prediction_sigma"][:, 3])
        gas_hard_masks.append(payload["prediction"][:, 3] <= (GAS_FLOOR_VALUE + 1e-8))

    pred_stack = torch.stack(predictions, dim=0)
    sigma_stack = torch.stack(prediction_sigma, dim=0)
    pred_mean = pred_stack.mean(dim=0)
    pred_std = _total_sigma(pred_stack, sigma_stack, final_mean=pred_mean)
    q_mean = torch.stack(quench_probs, dim=0).mean(dim=0)
    g_mean = torch.stack(gas_probs, dim=0).mean(dim=0)

    ssfr_pred_ens, ssfr_prob_ens = _majority_mean_from_hard(
        ssfr_point_preds,
        ssfr_hard_masks,
        floor_value=SSFR_QUENCH_THRESHOLD,
    )
    gas_pred_ens, gas_prob_ens = _majority_mean_from_hard(
        gas_point_preds,
        gas_hard_masks,
        floor_value=GAS_FLOOR_VALUE,
    )
    ssfr_sigma_ens = _total_sigma(
        torch.stack(ssfr_point_preds, dim=0),
        torch.stack(ssfr_point_sigma, dim=0),
        final_mean=ssfr_pred_ens,
    )
    gas_sigma_ens = _total_sigma(
        torch.stack(gas_point_preds, dim=0),
        torch.stack(gas_point_sigma, dim=0),
        final_mean=gas_pred_ens,
    )
    pred_mean[:, 4] = ssfr_pred_ens
    pred_std[:, 4] = ssfr_sigma_ens
    q_mean = ssfr_prob_ens
    pred_mean[:, 3] = gas_pred_ens
    pred_std[:, 3] = gas_sigma_ens
    g_mean = gas_prob_ens

    base = payload
    rows: list[dict[str, Any]] = []
    for i in range(pred_mean.shape[0]):
        row = {
            "node_index": int(base["node_index"][i]),
            "tree_id": int(base["tree_id"][i]),
            "z_index": int(base["z_index"][i]),
            "redshift": float(base["redshift"][i]),
            "quench_probability": float(q_mean[i]),
            "gas_floor_probability": float(g_mean[i]),
            "predicted_quenched": int(float(q_mean[i]) >= 0.5),
            "predicted_gas_floor": int(float(g_mean[i]) >= 0.5),
        }
        if selected_target_sam_id is not None:
            row["sam_id"] = int(selected_target_sam_id)
        for j, name in enumerate(TARGET_NAMES):
            row[name] = float(pred_mean[i, j])
            row[f"{name}_sigma"] = float(pred_std[i, j])
        rows.append(row)
    return {
        "mode": "ensemble",
        "artifact_name": artifact.name,
        "input_format": graph_input.source_format,
        "input_path": str(graph_input.source_path),
        "num_models": len(ckpts),
        "num_rows": len(rows),
        "rows": rows,
    }


def write_ensemble_prediction_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return output
