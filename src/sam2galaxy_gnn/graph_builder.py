from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json

import h5py
import numpy as np


@dataclass(frozen=True)
class GraphInput:
    source_format: str
    source_path: Path
    payload: Any


@dataclass(frozen=True)
class PredictionTargets:
    y_indices: Any
    z_indices: Any
    matched_z_values: list[float]


TARGET_Z_VALUES = [
    0.0004902500077150762,
    0.24828357994556427,
    0.4904018044471741,
    0.701982855796814,
    1.0319421291351318,
    1.5355613231658936,
    2.0292930603027344,
    2.9514758586883545,
    5.154603958129883,
]

HALO_HDF5_DATASETS = {
    "nodeIndex": "forestHalos/nodeIndex",
    "angularMomentum": "forestHalos/angularMomentum",
    "descendantIndex": "forestHalos/descendantIndex",
    "expansionFactor": "forestHalos/expansionFactor",
    "hostIndex": "forestHalos/hostIndex",
    "nodeMass": "forestHalos/nodeMass",
    "position": "forestHalos/position",
    "velocity": "forestHalos/velocity",
    "scaleRadius": "forestHalos/scaleRadius",
    "spin": "forestHalos/spin",
}


def _signed_log10(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    pos = values > 0
    neg = values < 0
    out[pos] = np.log10(values[pos])
    out[neg] = -np.log10(np.abs(values[neg]))
    return out


def _require_torch():
    import torch
    from torch_geometric.data import Data

    return torch, Data


def _load_tree_ids(tree_ids_path: str | Path) -> np.ndarray:
    values: list[int] = []
    with Path(tree_ids_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values.append(int(stripped))
    if not values:
        raise RuntimeError(f"No tree ids were found in {tree_ids_path}")
    return np.asarray(sorted(set(values)), dtype=np.int64)


def _load_node_index_to_tree_id_mapping(mapping_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    torch, _Data = _require_torch()
    payload = torch.load(Path(mapping_path), map_location="cpu")
    node_index = torch.as_tensor(payload["node_index"], dtype=torch.long).cpu().numpy().astype(np.int64)
    tree_id = torch.as_tensor(payload["tree_id"], dtype=torch.long).cpu().numpy().astype(np.int64)
    if node_index.shape != tree_id.shape:
        raise RuntimeError(f"node_index and tree_id shapes do not match in {mapping_path}")
    order = np.argsort(node_index)
    node_index = node_index[order]
    tree_id = tree_id[order]
    return node_index, tree_id


def load_graph_artifact(graph_path: str | Path, *, sam_parameters: Any | None = None) -> GraphInput:
    torch, _Data = _require_torch()
    path = Path(graph_path)
    graph = torch.load(path, map_location="cpu")
    if not hasattr(graph, "x"):
        raise RuntimeError(f"{path} is missing node feature tensor 'x'")
    if not hasattr(graph, "edge_index"):
        raise RuntimeError(f"{path} is missing edge tensor 'edge_index'")
    if sam_parameters is not None:
        graph.u = sam_parameters
    return GraphInput(source_format="graph", source_path=path, payload=graph)


def parse_halo_hdf5(hdf5_path: str | Path) -> GraphInput:
    path = Path(hdf5_path)
    with h5py.File(path, "r") as h5:
        missing = [name for name, ds in HALO_HDF5_DATASETS.items() if ds not in h5]
        count = int(h5[HALO_HDF5_DATASETS["nodeIndex"]].shape[0]) if not missing else 0
    if missing:
        raise RuntimeError(f"Missing required halo HDF5 datasets {missing} in {path}")
    return GraphInput(source_format="halo_hdf5", source_path=path, payload={"validated": True, "num_nodes": count})


def build_graph_from_halo_hdf5(
    hdf5_path: str | Path,
    *,
    sam_parameters: Any | None = None,
    tree_ids_path: str | Path | None = None,
    node_index_to_tree_id_path: str | Path | None = None,
) -> GraphInput:
    torch, Data = _require_torch()
    path = Path(hdf5_path)
    with h5py.File(path, "r") as h5:
        node_index = np.asarray(h5[HALO_HDF5_DATASETS["nodeIndex"]][:], dtype=np.int64)
        angular_momentum = np.asarray(h5[HALO_HDF5_DATASETS["angularMomentum"]][:], dtype=np.float32)
        descendant_index = np.asarray(h5[HALO_HDF5_DATASETS["descendantIndex"]][:], dtype=np.int64)
        expansion_factor = np.asarray(h5[HALO_HDF5_DATASETS["expansionFactor"]][:], dtype=np.float32)
        host_index = np.asarray(h5[HALO_HDF5_DATASETS["hostIndex"]][:], dtype=np.int64)
        node_mass = np.asarray(h5[HALO_HDF5_DATASETS["nodeMass"]][:], dtype=np.float32)
        position = np.asarray(h5[HALO_HDF5_DATASETS["position"]][:], dtype=np.float32)
        velocity = np.asarray(h5[HALO_HDF5_DATASETS["velocity"]][:], dtype=np.float32)
        scale_radius = np.asarray(h5[HALO_HDF5_DATASETS["scaleRadius"]][:], dtype=np.float32)
        spin = np.asarray(h5[HALO_HDF5_DATASETS["spin"]][:], dtype=np.float32)
    selected_tree_ids = None
    tree_id_lookup = None
    if tree_ids_path is not None:
        if node_index_to_tree_id_path is None:
            raise RuntimeError("tree_ids_path requires node_index_to_tree_id_path")
        selected_tree_ids = _load_tree_ids(tree_ids_path)
        mapped_node_index, mapped_tree_id = _load_node_index_to_tree_id_mapping(node_index_to_tree_id_path)
        loc = np.searchsorted(mapped_node_index, node_index)
        exact = np.zeros(node_index.shape[0], dtype=bool)
        in_bounds = loc < mapped_node_index.shape[0]
        exact[in_bounds] = mapped_node_index[loc[in_bounds]] == node_index[in_bounds]
        node_tree_id = np.full(node_index.shape[0], -1, dtype=np.int64)
        node_tree_id[exact] = mapped_tree_id[loc[exact]]
        keep = np.isin(node_tree_id, selected_tree_ids)
        if not np.any(keep):
            raise RuntimeError(f"No halo HDF5 rows matched the requested tree ids from {tree_ids_path}")
        node_index = node_index[keep]
        angular_momentum = angular_momentum[keep]
        descendant_index = descendant_index[keep]
        expansion_factor = expansion_factor[keep]
        host_index = host_index[keep]
        node_mass = node_mass[keep]
        position = position[keep]
        velocity = velocity[keep]
        scale_radius = scale_radius[keep]
        spin = spin[keep]
        node_tree_id = node_tree_id[keep]
    else:
        node_tree_id = np.zeros(node_index.shape[0], dtype=np.int64)

    redshift = 1.0 / np.clip(expansion_factor, 1e-8, None) - 1.0
    x = np.stack(
        [
            _signed_log10(angular_momentum[:, 0]),
            _signed_log10(angular_momentum[:, 1]),
            _signed_log10(angular_momentum[:, 2]),
            _signed_log10(node_mass),
            position[:, 0],
            position[:, 1],
            position[:, 2],
            velocity[:, 0],
            velocity[:, 1],
            velocity[:, 2],
            scale_radius,
            spin,
            expansion_factor,
            redshift.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    id_to_local = {int(idx): i for i, idx in enumerate(node_index.tolist())}
    edge_src: list[int] = []
    edge_dst: list[int] = []
    for idx, (node_id, desc_id, host_id) in enumerate(zip(node_index, descendant_index, host_index)):
        edge_src.append(idx)
        edge_dst.append(idx)
        if int(desc_id) in id_to_local and int(desc_id) != int(node_id):
            edge_src.append(idx)
            edge_dst.append(id_to_local[int(desc_id)])
        if int(host_id) in id_to_local and int(host_id) != int(node_id):
            edge_src.append(idx)
            edge_dst.append(id_to_local[int(host_id)])

    graph = Data(
        x=torch.from_numpy(x),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        tree_id=torch.from_numpy(node_tree_id.astype(np.int64)),
        node_id=torch.from_numpy(node_index.astype(np.int64)),
    )
    # The public HDF5 route is inference-only. It reconstructs node features and
    # connectivity but does not attach labels, targets, or any train-time cache fields.
    if sam_parameters is not None:
        graph.u = sam_parameters
    return GraphInput(source_format="halo_hdf5", source_path=path, payload=graph)


def select_prediction_targets(graph_input: GraphInput) -> PredictionTargets:
    torch, _Data = _require_torch()
    if graph_input.source_format not in {"graph", "halo_hdf5"} or not hasattr(graph_input.payload, "x"):
        raise RuntimeError("Prediction targets require a graph payload")
    graph = graph_input.payload
    unique_z = torch.unique(graph.x[:, 13]).cpu().tolist()
    matched = []
    for target_z in TARGET_Z_VALUES:
        best = min(unique_z, key=lambda val: abs(float(val) - float(target_z)))
        matched.append(float(best))
    y_idx_parts = []
    z_idx_parts = []
    for z_i, z_val in enumerate(matched):
        mask = torch.isclose(graph.x[:, 13], torch.tensor(z_val, dtype=graph.x.dtype), atol=1e-6)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        y_idx_parts.append(idx)
        z_idx_parts.append(torch.full((idx.numel(),), z_i, dtype=torch.long))
    if not y_idx_parts:
        raise RuntimeError("No nodes matched the public prediction redshift layers")
    return PredictionTargets(
        y_indices=torch.cat(y_idx_parts, dim=0),
        z_indices=torch.cat(z_idx_parts, dim=0),
        matched_z_values=matched,
    )


def load_selected_targets(
    graph_input: GraphInput,
    selected_targets_path: str | Path,
    *,
    sam_id: int | None = None,
) -> PredictionTargets:
    torch, _Data = _require_torch()
    graph = graph_input.payload
    if hasattr(graph, "global_node_index"):
        graph_keys = torch.as_tensor(graph.global_node_index, dtype=torch.long)
    elif hasattr(graph, "node_id"):
        graph_keys = torch.as_tensor(graph.node_id, dtype=torch.long)
    else:
        graph_keys = torch.arange(graph.num_nodes, dtype=torch.long)

    key_to_local = {int(graph_keys[i]): int(i) for i in range(graph_keys.numel())}
    y_idx_parts = []
    z_idx_parts = []
    matched = sorted({float(z) for z in torch.unique(graph.x[:, 13]).cpu().tolist()})
    with Path(selected_targets_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if sam_id is not None and "sam_id" in row and row["sam_id"] != "":
                if int(row["sam_id"]) != int(sam_id):
                    continue
            node_index = int(row["node_index"])
            z_index = int(row["z_index"])
            local_idx = key_to_local.get(node_index)
            if local_idx is None:
                continue
            y_idx_parts.append(local_idx)
            z_idx_parts.append(z_index)
    if not y_idx_parts:
        raise RuntimeError(f"No target-indicator rows from {selected_targets_path} matched the graph payload")
    return PredictionTargets(
        y_indices=torch.tensor(y_idx_parts, dtype=torch.long),
        z_indices=torch.tensor(z_idx_parts, dtype=torch.long),
        matched_z_values=matched,
    )


def summarize_graph_input(graph_input: GraphInput) -> dict[str, Any]:
    if graph_input.source_format in {"graph", "halo_hdf5"} and hasattr(graph_input.payload, "x"):
        graph = graph_input.payload
        return {
            "input_format": graph_input.source_format,
            "source_path": str(graph_input.source_path),
            "num_nodes": int(graph.x.shape[0]),
            "num_node_features": int(graph.x.shape[1]),
            "num_edges": int(graph.edge_index.shape[1]),
            "has_tree_id": bool(hasattr(graph, "tree_id")),
            "has_sam_parameters": bool(hasattr(graph, "u")),
        }
    if graph_input.source_format == "halo_hdf5":
        return {
            "input_format": "halo_hdf5",
            "source_path": str(graph_input.source_path),
            "num_nodes": int(graph_input.payload.get("num_nodes", 0)),
            "required_datasets": sorted(HALO_HDF5_DATASETS.keys()),
        }
    raise RuntimeError(f"Unsupported graph input summary route for {graph_input.source_format}")


def write_graph_summary(graph_input: GraphInput, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(summarize_graph_input(graph_input), fh, indent=2)
    return output
