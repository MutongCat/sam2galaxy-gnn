from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import csv

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
import yaml

from ._vendor.model_sam_gnn import SAMGalaxyGNN_MultiZ


TARGET_NAMES = [
    "stellar_mass",
    "sdss_z_band_luminosity",
    "angular_momentum",
    "gas_metal_mass",
    "specific_star_formation_rate",
]


@dataclass(frozen=True)
class ScalerStats:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    u_mean: torch.Tensor
    u_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


class InferenceBatch(Data):
    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "y_indices":
            return self.num_nodes
        if key == "z_indices":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def _std_from_state(state: dict[str, Any], device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=device)
    m2 = torch.as_tensor(state["M2"], dtype=torch.float32, device=device)
    n = torch.as_tensor(state["n"], dtype=torch.float32, device=device)
    denom = torch.clamp(n - 1.0, min=1.0)
    return torch.sqrt(m2 / denom).clamp_min(1e-8)


def load_scaler_stats(scaler_path: str | Path, *, device: torch.device) -> ScalerStats:
    payload = torch.load(Path(scaler_path), map_location="cpu")
    x_state = payload["scaler_x"]
    u_state = payload["scaler_u"]
    y_state = payload["scaler_y"]
    return ScalerStats(
        x_mean=torch.as_tensor(x_state["mean"], dtype=torch.float32, device=device),
        x_std=_std_from_state(x_state, device),
        u_mean=torch.as_tensor(u_state["mean"], dtype=torch.float32, device=device),
        u_std=_std_from_state(u_state, device),
        y_mean=torch.as_tensor(y_state["mean"], dtype=torch.float32, device=device),
        y_std=_std_from_state(y_state, device),
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_sam_parameters_json(path: str | Path) -> torch.Tensor:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get("values", payload.get("sam_parameters"))
    else:
        values = payload
    if not isinstance(values, list) or len(values) != 17:
        raise ValueError(f"Expected 17 SAM parameters in {path}")
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def _load_sam_parameters_dat(path: str | Path, *, sam_id: int | None = None) -> torch.Tensor:
    rows: list[tuple[int, list[float]]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        first = fh.read(4096)
        fh.seek(0)
        dialect = csv.excel_tab if "\t" in first else csv.excel
        reader = csv.reader(fh, dialect=dialect)
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                row = [part for part in row[0].replace(",", " ").split() if part]
            row = [cell.strip() for cell in row if str(cell).strip()]
            if not row or row[0].startswith("#"):
                continue
            try:
                row_sam_id = int(row[0])
                values = [float(x) for x in row[1:]]
            except ValueError:
                continue
            if len(values) != 17:
                raise ValueError(f"Expected 17 SAM parameters after sam_id in {path}")
            rows.append((row_sam_id, values))
    if not rows:
        raise ValueError(f"No SAM parameter rows found in {path}")
    if sam_id is None:
        values = rows[0][1]
    else:
        matched = [values for row_sam_id, values in rows if row_sam_id == int(sam_id)]
        if not matched:
            raise ValueError(f"sam_id={sam_id} was not found in {path}")
        values = matched[0]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def load_sam_parameters(path: str | Path, *, sam_id: int | None = None) -> torch.Tensor:
    input_path = Path(path)
    if input_path.suffix.lower() == ".json":
        return _load_sam_parameters_json(input_path)
    if input_path.suffix.lower() == ".dat":
        return _load_sam_parameters_dat(input_path, sam_id=sam_id)
    raise ValueError(f"Unsupported SAM parameter file format for {path}")


def build_model_from_config(config_path: str | Path, checkpoint_path: str | Path, *, device: torch.device) -> SAMGalaxyGNN_MultiZ:
    cfg = load_yaml(config_path)
    model_cfg = cfg["model"]
    model = SAMGalaxyGNN_MultiZ(
        in_dim=model_cfg.get("in_dim", 14),
        u_dim=model_cfg.get("u_dim", 17),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        out_dim_per_z=model_cfg.get("out_dim_per_z", 5),
        num_z=model_cfg.get("num_z", 9),
        use_moe=bool(model_cfg.get("use_moe", True)),
        mix_hidden_dim=model_cfg.get("mix_hidden_dim", 128),
        mix_hidden_layers=model_cfg.get("mix_hidden_layers", 2),
        mix_dropout=model_cfg.get("mix_dropout", 0.2),
        ssfr_mix_components=model_cfg.get("ssfr_mix_components", 3),
        ssfr_mix_continuous=model_cfg.get("ssfr_mix_continuous", 2),
        gas_mix_components=model_cfg.get("gas_mix_components", 4),
        gas_mix_continuous=model_cfg.get("gas_mix_continuous", 3),
        reg_head_type=model_cfg.get("reg_head_type", "hetero"),
        reg_sigma_min=model_cfg.get("reg_sigma_min", -7.0),
        reg_sigma_max=model_cfg.get("reg_sigma_max", 7.0),
    ).to(device)
    model.load_state_dict(torch.load(Path(checkpoint_path), map_location=device))
    model.eval()
    return model


def _mix_point_est_with_prob(
    mix_tuple: Any,
    *,
    floor_norm: float,
    floor_sigma_phys: float,
    std_norm: float,
    point_estimate: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mix_tuple is None:
        return None, None
    mix_logits, mu, log_sigma = mix_tuple
    weights = torch.softmax(mix_logits, dim=-1)
    floor_prob = weights[:, 0]
    if point_estimate == "mean":
        cont = (weights[:, 1:] * mu).sum(dim=-1)
        pred = weights[:, 0] * floor_norm + cont
        return pred, floor_prob
    sigma_floor = max(1e-6, float(floor_sigma_phys)) / float(std_norm)
    log_pi = torch.log(weights.clamp_min(1e-12))
    log_score0 = log_pi[:, 0] - torch.log(torch.tensor(sigma_floor, dtype=mu.dtype, device=mu.device))
    log_score_cont = log_pi[:, 1:] - log_sigma
    all_scores = torch.cat([log_score0.unsqueeze(1), log_score_cont], dim=1)
    k = torch.argmax(all_scores, dim=-1)
    pred = mu.gather(1, (k.clamp_min(1) - 1).view(-1, 1)).squeeze(1)
    pred = torch.where(k == 0, torch.full_like(pred, floor_norm), pred)
    return pred, floor_prob


def _mix_predictive_sigma(
    mix_tuple: Any,
    *,
    floor_norm: float,
    floor_sigma_phys: float,
    std_norm: float,
) -> torch.Tensor | None:
    if mix_tuple is None:
        return None
    mix_logits, mu, log_sigma = mix_tuple
    weights = torch.softmax(mix_logits, dim=-1)
    sigma_floor = max(1e-6, float(floor_sigma_phys)) / float(std_norm)
    floor_mean = torch.full(
        (weights.shape[0], 1),
        float(floor_norm),
        dtype=mu.dtype,
        device=mu.device,
    )
    floor_var = torch.full_like(floor_mean, float(sigma_floor) ** 2)
    cont_var = torch.exp(2.0 * log_sigma)
    comp_mean = torch.cat([floor_mean, mu], dim=-1)
    comp_var = torch.cat([floor_var, cont_var], dim=-1)
    second_moment = (weights * (comp_var + comp_mean.square())).sum(dim=-1)
    mean = (weights * comp_mean).sum(dim=-1)
    variance = (second_moment - mean.square()).clamp_min(0.0)
    return torch.sqrt(variance)


def build_inference_batches(
    graph: Data,
    *,
    y_indices: torch.Tensor,
    z_indices: torch.Tensor,
    matched_z_values: list[float],
    max_trees_per_batch: int = 128,
) -> list[InferenceBatch]:
    tree_ids = torch.as_tensor(graph.tree_id, dtype=torch.long)
    if hasattr(graph, "global_node_index"):
        output_node_index = torch.as_tensor(graph.global_node_index, dtype=torch.long)
    elif hasattr(graph, "node_id"):
        output_node_index = torch.as_tensor(graph.node_id, dtype=torch.long)
    else:
        output_node_index = torch.arange(graph.num_nodes, dtype=torch.long)
    unique_tree_ids = torch.unique(tree_ids)
    graph_node_ids = torch.arange(graph.num_nodes, dtype=torch.long)
    batches: list[InferenceBatch] = []
    for start in range(0, unique_tree_ids.numel(), max_trees_per_batch):
        batch_tree_ids = unique_tree_ids[start : start + max_trees_per_batch]
        keep_nodes = torch.isin(tree_ids, batch_tree_ids)
        node_idx = graph_node_ids[keep_nodes]
        edge_index_sub, _ = subgraph(node_idx, graph.edge_index, relabel_nodes=True)
        global_to_local = torch.full((graph.num_nodes,), -1, dtype=torch.long)
        global_to_local[node_idx] = torch.arange(node_idx.numel(), dtype=torch.long)
        keep_targets = keep_nodes[y_indices]
        if not bool(keep_targets.any()):
            continue
        local_y = global_to_local[y_indices[keep_targets]]
        local_z = z_indices[keep_targets]
        sub = InferenceBatch(
            x=graph.x[node_idx],
            edge_index=edge_index_sub,
            u=graph.u,
            y_indices=local_y,
            z_indices=local_z,
            node_index_global_y=output_node_index[y_indices[keep_targets]],
            tree_id_y=tree_ids[y_indices[keep_targets]],
            redshift_y=graph.x[y_indices[keep_targets], 13],
        )
        sub.target_redshift_values = torch.tensor(matched_z_values, dtype=torch.float32)
        batches.append(sub)
    return batches


def run_model_on_batches(
    model: SAMGalaxyGNN_MultiZ,
    batches: list[InferenceBatch],
    *,
    scaler: ScalerStats,
    point_estimate: str = "map",
    ssfr_point_sigma: float = 0.15,
    gas_point_sigma: float = 0.05,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    preds: list[torch.Tensor] = []
    node_ids: list[torch.Tensor] = []
    tree_ids: list[torch.Tensor] = []
    z_indices_out: list[torch.Tensor] = []
    redshifts: list[torch.Tensor] = []
    ssfr_prob: list[torch.Tensor] = []
    gas_prob: list[torch.Tensor] = []
    pred_sigma: list[torch.Tensor] = []

    loader = DataLoader(batches, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, "batch") or batch.batch is None:
                batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            batch.x = (batch.x - scaler.x_mean) / scaler.x_std
            batch.u = (batch.u - scaler.u_mean) / scaler.u_std
            out = model(batch.x, batch.edge_index, batch.u, batch, batch.z_indices)

            reg_rows: list[torch.Tensor] = []
            sigma_rows: list[torch.Tensor] = []
            qprob_rows: list[torch.Tensor] = []
            gprob_rows: list[torch.Tensor] = []
            for z_i in range(model.num_z):
                sel = batch.z_indices == z_i
                if not bool(sel.any()):
                    continue
                pred_dict = out[f"z{z_i}"]
                reg = pred_dict["reg"]
                reg_sigma = pred_dict.get("log_sigma")
                mix = pred_dict.get("mix", {})
                reg_local = reg.clone()

                qprob = torch.zeros(reg.size(0), dtype=reg.dtype, device=device)
                gprob = torch.zeros(reg.size(0), dtype=reg.dtype, device=device)
                if mix:
                    ssfr_floor_norm = float((-2.0 - scaler.y_mean[4]) / scaler.y_std[4])
                    gas_floor_norm = float((-5.0 - scaler.y_mean[3]) / scaler.y_std[3])
                    ssfr_pred, ssfr_floor_prob = _mix_point_est_with_prob(
                        mix.get("ssfr"),
                        floor_norm=ssfr_floor_norm,
                        floor_sigma_phys=ssfr_point_sigma,
                        std_norm=float(scaler.y_std[4]),
                        point_estimate=point_estimate,
                    )
                    gas_pred, gas_floor_prob = _mix_point_est_with_prob(
                        mix.get("gas"),
                        floor_norm=gas_floor_norm,
                        floor_sigma_phys=gas_point_sigma,
                        std_norm=float(scaler.y_std[3]),
                        point_estimate=point_estimate,
                    )
                    if ssfr_pred is not None:
                        reg_local[:, 4] = ssfr_pred
                        qprob = ssfr_floor_prob
                    if gas_pred is not None:
                        reg_local[:, 3] = gas_pred
                        gprob = gas_floor_prob

                reg_phys = reg_local * scaler.y_std.unsqueeze(0) + scaler.y_mean.unsqueeze(0)
                reg_rows.append(reg_phys)
                qprob_rows.append(qprob)
                gprob_rows.append(gprob)
                if reg_sigma is not None:
                    sigma_phys = torch.exp(reg_sigma) * scaler.y_std.unsqueeze(0)
                    if mix:
                        ssfr_mix_sigma = _mix_predictive_sigma(
                            mix.get("ssfr"),
                            floor_norm=ssfr_floor_norm,
                            floor_sigma_phys=ssfr_point_sigma,
                            std_norm=float(scaler.y_std[4]),
                        )
                        gas_mix_sigma = _mix_predictive_sigma(
                            mix.get("gas"),
                            floor_norm=gas_floor_norm,
                            floor_sigma_phys=gas_point_sigma,
                            std_norm=float(scaler.y_std[3]),
                        )
                        if ssfr_mix_sigma is not None:
                            sigma_phys[:, 4] = ssfr_mix_sigma * scaler.y_std[4]
                        if gas_mix_sigma is not None:
                            sigma_phys[:, 3] = gas_mix_sigma * scaler.y_std[3]
                    sigma_rows.append(sigma_phys)
                else:
                    sigma_rows.append(torch.full_like(reg_phys, float("nan")))

            preds.append(torch.cat(reg_rows, dim=0).cpu())
            pred_sigma.append(torch.cat(sigma_rows, dim=0).cpu())
            ssfr_prob.append(torch.cat(qprob_rows, dim=0).cpu())
            gas_prob.append(torch.cat(gprob_rows, dim=0).cpu())
            node_ids.append(batch.node_index_global_y.cpu())
            tree_ids.append(batch.tree_id_y.cpu())
            z_indices_out.append(batch.z_indices.cpu())
            redshifts.append(batch.redshift_y.cpu())

    return {
        "node_index": torch.cat(node_ids, dim=0),
        "tree_id": torch.cat(tree_ids, dim=0),
        "z_index": torch.cat(z_indices_out, dim=0),
        "redshift": torch.cat(redshifts, dim=0),
        "prediction": torch.cat(preds, dim=0),
        "prediction_sigma": torch.cat(pred_sigma, dim=0),
        "quench_probability": torch.cat(ssfr_prob, dim=0),
        "gas_floor_probability": torch.cat(gas_prob, dim=0),
    }
