#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam2galaxy_gnn.artifacts import load_release_artifacts
from sam2galaxy_gnn.example_eval import (
    align_prediction_rows,
    build_joined_eval,
    load_csv_rows,
    write_summary_json,
)
from sam2galaxy_gnn.graph_builder import load_graph_artifact
from sam2galaxy_gnn.predict_ensemble import predict_ensemble
from sam2galaxy_gnn.predict_single import predict_single
from sam2galaxy_gnn.runtime import load_sam_parameters
from sam2galaxy_gnn.export import write_prediction_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--target-indicator", required=True)
    parser.add_argument("--truth-values", required=True)
    parser.add_argument("--sam-parameters", required=True)
    parser.add_argument("--output-single-dir", required=True)
    parser.add_argument("--output-ensemble-dir", required=True)
    parser.add_argument("--output-single-summary", required=True)
    parser.add_argument("--output-ensemble-summary", required=True)
    args = parser.parse_args()

    artifacts = load_release_artifacts(args.manifest)
    target_rows = load_csv_rows(args.target_indicator)
    truth_rows = load_csv_rows(args.truth_values)
    sam_ids = sorted({int(row["sam_id"]) for row in truth_rows})

    single_rows = []
    ensemble_rows = []
    for sam_id in sam_ids:
        sam_params = load_sam_parameters(args.sam_parameters, sam_id=sam_id)
        graph_input = load_graph_artifact(args.graph, sam_parameters=sam_params)
        single_summary = predict_single(
            graph_input,
            artifacts.single_model,
            selected_targets_path=args.target_indicator,
            selected_target_sam_id=sam_id,
        )
        ensemble_summary = predict_ensemble(
            graph_input,
            artifacts.ensemble_model,
            selected_targets_path=args.target_indicator,
            selected_target_sam_id=sam_id,
        )
        single_sam_rows = align_prediction_rows(
            prediction_rows=single_summary["rows"],
            selected_target_rows=target_rows,
        )
        ensemble_sam_rows = align_prediction_rows(
            prediction_rows=ensemble_summary["rows"],
            selected_target_rows=target_rows,
        )
        single_rows.extend(single_sam_rows)
        ensemble_rows.extend(ensemble_sam_rows)
        write_prediction_rows(single_sam_rows, Path(args.output_single_dir) / f"sam_{sam_id:04d}_predictions.csv")
        write_prediction_rows(ensemble_sam_rows, Path(args.output_ensemble_dir) / f"sam_{sam_id:04d}_predictions.csv")

    single_rows = align_prediction_rows(
        prediction_rows=single_rows,
        selected_target_rows=target_rows,
    )
    ensemble_rows = align_prediction_rows(
        prediction_rows=ensemble_rows,
        selected_target_rows=target_rows,
    )
    _single_joined, single_summary = build_joined_eval(
        prediction_rows=single_rows,
        truth_rows=truth_rows,
    )
    _ensemble_joined, ensemble_summary = build_joined_eval(
        prediction_rows=ensemble_rows,
        truth_rows=truth_rows,
    )

    write_summary_json(single_summary, args.output_single_summary)
    write_summary_json(ensemble_summary, args.output_ensemble_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
