from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ReleaseConfig, load_release_config


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    config_path: Path
    scaler_path: Path
    checkpoint_path: Path | None = None
    checkpoint_list_path: Path | None = None
    models_dir: Path | None = None


@dataclass(frozen=True)
class ExampleInputs:
    graph_path: Path
    halo_hdf5_path: Path | None = None
    node_index_to_tree_id_path: Path | None = None
    tree_ids_path: Path | None = None
    sam_parameters_path: Path | None = None
    target_indicator_path: Path | None = None
    truth_path: Path | None = None


@dataclass(frozen=True)
class ReleaseArtifacts:
    config: ReleaseConfig
    single_model: ModelArtifact
    ensemble_model: ModelArtifact
    example_inputs: ExampleInputs
    output_targets: list[str]


def _to_model_artifact(payload: dict[str, Any]) -> ModelArtifact:
    return ModelArtifact(
        name=str(payload["name"]),
        config_path=Path(payload["config_path"]),
        scaler_path=Path(payload["scaler_path"]),
        checkpoint_path=Path(payload["checkpoint_path"]) if payload.get("checkpoint_path") else None,
        checkpoint_list_path=Path(payload["checkpoint_list_path"]) if payload.get("checkpoint_list_path") else None,
        models_dir=Path(payload["models_dir"]) if payload.get("models_dir") else None,
    )


def _resolve_path(cfg: ReleaseConfig, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (cfg.manifest_dir / path).resolve()


def load_release_artifacts(manifest_path: str | Path | None = None) -> ReleaseArtifacts:
    cfg = load_release_config(manifest_path=manifest_path)
    manifest = cfg.manifest
    single = manifest["single_model"].copy()
    ensemble = manifest["ensemble_model"].copy()
    examples = manifest["example_inputs"].copy()
    for payload in [single, ensemble]:
        for key in ["config_path", "scaler_path", "checkpoint_path", "checkpoint_list_path", "models_dir"]:
            payload[key] = str(_resolve_path(cfg, payload.get(key))) if payload.get(key) else None
    return ReleaseArtifacts(
        config=cfg,
        single_model=_to_model_artifact(single),
        ensemble_model=_to_model_artifact(ensemble),
        example_inputs=ExampleInputs(
            halo_hdf5_path=_resolve_path(cfg, examples.get("halo_hdf5_path")),
            node_index_to_tree_id_path=_resolve_path(cfg, examples.get("node_index_to_tree_id_path")),
            graph_path=_resolve_path(cfg, examples["graph_path"]),
            tree_ids_path=_resolve_path(cfg, examples.get("tree_ids_path")),
            sam_parameters_path=_resolve_path(cfg, examples.get("sam_parameters_path")),
            target_indicator_path=_resolve_path(cfg, examples.get("target_indicator_path")),
            truth_path=_resolve_path(cfg, examples.get("truth_path")),
        ),
        output_targets=list(manifest["output_targets"]),
    )


def validate_artifacts(artifacts: ReleaseArtifacts) -> list[str]:
    missing: list[str] = []
    for path in [
        artifacts.single_model.config_path,
        artifacts.single_model.scaler_path,
        artifacts.ensemble_model.config_path,
        artifacts.ensemble_model.scaler_path,
        artifacts.example_inputs.graph_path,
    ]:
        if not path.exists():
            missing.append(str(path))
    for path in [
        artifacts.example_inputs.sam_parameters_path,
        artifacts.example_inputs.node_index_to_tree_id_path,
        artifacts.example_inputs.tree_ids_path,
        artifacts.example_inputs.truth_path,
        artifacts.example_inputs.target_indicator_path,
    ]:
        if path is not None and not path.exists():
            missing.append(str(path))
    if artifacts.single_model.checkpoint_path and not artifacts.single_model.checkpoint_path.exists():
        missing.append(str(artifacts.single_model.checkpoint_path))
    if artifacts.ensemble_model.checkpoint_list_path and not artifacts.ensemble_model.checkpoint_list_path.exists():
        missing.append(str(artifacts.ensemble_model.checkpoint_list_path))
    return missing
