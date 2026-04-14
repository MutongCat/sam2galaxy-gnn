from pathlib import Path
import csv
import json

import pytest

from sam2galaxy_gnn.artifacts import load_release_artifacts
from sam2galaxy_gnn.graph_builder import parse_halo_hdf5, summarize_graph_input


def test_manifest_loads():
    artifacts = load_release_artifacts()
    assert artifacts.output_targets
    assert artifacts.single_model.name
    assert artifacts.ensemble_model.name
    assert artifacts.example_inputs.halo_hdf5_path is not None
    assert artifacts.example_inputs.node_index_to_tree_id_path is not None
    assert artifacts.example_inputs.tree_ids_path is not None
    assert artifacts.example_inputs.sam_parameters_path is not None
    assert artifacts.example_inputs.target_indicator_path is not None
    assert artifacts.example_inputs.truth_path is not None

def test_example_eval_keys_match():
    artifacts = load_release_artifacts()
    with artifacts.example_inputs.target_indicator_path.open("r", encoding="utf-8", newline="") as fh:
        target_rows = list(csv.DictReader(fh))
    with artifacts.example_inputs.truth_path.open("r", encoding="utf-8", newline="") as fh:
        truth = list(csv.DictReader(fh))
    selected_keys = {(row["sam_id"], row["node_index"], row["z_index"]) for row in target_rows}
    truth_keys = {(row["sam_id"], row["node_index"], row["z_index"]) for row in truth}
    assert selected_keys
    assert selected_keys == truth_keys


def test_example_sam_table_has_eight_rows_and_fixed_width():
    artifacts = load_release_artifacts()
    rows = []
    with artifacts.example_inputs.sam_parameters_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        rows = [row for row in reader if row]
    assert header[0] == "sam_id"
    assert len(header) == 18
    assert len(rows) == 8
    assert all(len(row) == 18 for row in rows)


def test_halo_hdf5_summary():
    artifacts = load_release_artifacts()
    if artifacts.example_inputs.halo_hdf5_path is None or not artifacts.example_inputs.halo_hdf5_path.exists():
        pytest.skip("external halo HDF5 not downloaded")
    graph_input = parse_halo_hdf5(artifacts.example_inputs.halo_hdf5_path)
    summary = summarize_graph_input(graph_input)
    assert summary["input_format"] == "halo_hdf5"
    assert summary["num_nodes"] > 0


def test_reference_tree_ids_match_eval_manifest():
    root = Path(__file__).resolve().parents[1]
    tree_ids_path = root / "examples" / "input" / "tree_input" / "tree_ids.txt"
    inference_tree_ids = [int(line.strip()) for line in tree_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(inference_tree_ids) == 1024


def test_eval_summary_uses_continuous_masks():
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "examples" / "output" / "single_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["coverage_fraction"] == 1.0
    assert "regression_mode" not in summary["targets"]["gas_metal_mass"]
    assert "regression_mode" not in summary["targets"]["specific_star_formation_rate"]
    assert "floor_collapse_diagnostic" not in summary["targets"]["gas_metal_mass"]
    assert summary["targets"]["gas_metal_mass"]["regression_row_count"] < summary["num_joined_rows"]
    assert summary["targets"]["specific_star_formation_rate"]["regression_row_count"] < summary["num_joined_rows"]


def test_per_sam_prediction_catalogs_exist_for_both_model_families():
    root = Path(__file__).resolve().parents[1]
    sam_table_path = root / "examples" / "input" / "graph_input" / "sam_parameters.dat"
    with sam_table_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        sam_ids = [int(row[0]) for row in reader if row]
    single_dir = root / "examples" / "output" / "single"
    ensemble_dir = root / "examples" / "output" / "ensemble"
    single_files = sorted(path.name for path in single_dir.glob("sam_*_predictions.csv"))
    ensemble_files = sorted(path.name for path in ensemble_dir.glob("sam_*_predictions.csv"))
    expected = sorted(f"sam_{sam_id:04d}_predictions.csv" for sam_id in sam_ids)
    assert single_files == expected
    assert ensemble_files == expected


def test_example_input_subdirectories_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples" / "input" / "tree_input").is_dir()
    assert (root / "examples" / "input" / "graph_input").is_dir()
