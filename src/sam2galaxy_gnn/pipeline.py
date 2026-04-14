from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import load_release_artifacts
from .graph_builder import build_graph_from_halo_hdf5, load_graph_artifact
from .predict_single import predict_single
from .predict_ensemble import predict_ensemble
from .runtime import load_sam_parameters


def run_pipeline(
    *,
    input_format: str,
    input_path: str | Path,
    model_type: str,
    manifest_path: str | Path | None = None,
    sam_parameters_path: str | Path | None = None,
    sam_id: int | None = None,
    tree_ids_path: str | Path | None = None,
    target_indicator_path: str | Path | None = None,
    target_indicator_sam_id: int | None = None,
) -> dict[str, Any]:
    artifacts = load_release_artifacts(manifest_path=manifest_path)
    sam_params = None
    chosen_sam_path = sam_parameters_path or artifacts.example_inputs.sam_parameters_path
    if chosen_sam_path is not None:
        sam_params = load_sam_parameters(chosen_sam_path, sam_id=sam_id)

    if input_format == "graph":
        graph_input = load_graph_artifact(input_path, sam_parameters=sam_params)
    elif input_format == "halo_hdf5":
        chosen_tree_ids_path = tree_ids_path or artifacts.example_inputs.tree_ids_path
        graph_input = build_graph_from_halo_hdf5(
            input_path,
            sam_parameters=sam_params,
            tree_ids_path=chosen_tree_ids_path,
            node_index_to_tree_id_path=artifacts.example_inputs.node_index_to_tree_id_path,
        )
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    if model_type == "single":
        return predict_single(
            graph_input,
            artifacts.single_model,
            selected_targets_path=target_indicator_path,
            selected_target_sam_id=(target_indicator_sam_id if target_indicator_sam_id is not None else sam_id),
        )
    if model_type == "ensemble":
        return predict_ensemble(
            graph_input,
            artifacts.ensemble_model,
            selected_targets_path=target_indicator_path,
            selected_target_sam_id=(target_indicator_sam_id if target_indicator_sam_id is not None else sam_id),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
