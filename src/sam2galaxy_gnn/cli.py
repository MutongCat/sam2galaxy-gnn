from __future__ import annotations

import argparse
from pathlib import Path
import json

from .artifacts import load_release_artifacts, validate_artifacts
from .export import write_prediction_rows
from .graph_builder import load_graph_artifact, parse_halo_hdf5, write_graph_summary
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam2galaxy-gnn")
    parser.add_argument("--manifest", default=None, help="Optional release manifest path.")
    parser.add_argument("--input-format", choices=["halo_hdf5", "graph"], required=True)
    parser.add_argument("--input", required=True, help="Input halo HDF5 or graph path.")
    parser.add_argument("--output", required=True, help="Output summary path.")
    parser.add_argument("--model-type", choices=["single", "ensemble"], default="single")
    parser.add_argument("--sam-parameters", default=None, help="Optional SAM-parameter JSON or DAT file.")
    parser.add_argument("--sam-id", type=int, default=None, help="Optional sam_id used to select one row from a multi-SAM DAT file.")
    parser.add_argument(
        "--tree-ids-file",
        default=None,
        help="Optional text file with one tree_id per line for halo-HDF5 tree selection.",
    )
    parser.add_argument(
        "--target-indicator",
        default=None,
        help="Optional target-row CSV used to override the default all-halos-on-output-layers target rule.",
    )
    parser.add_argument(
        "--target-indicator-sam-id",
        type=int,
        default=None,
        help="Optional SAM id used to filter target-indicator CSV rows when the CSV contains a sam_id column.",
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "predict"],
        default="summary",
        help="Write an input summary or a prediction summary.",
    )
    parser.add_argument(
        "--validate-artifacts",
        action="store_true",
        help="Validate configured release artifact paths before running.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.validate_artifacts:
        artifacts = load_release_artifacts(args.manifest)
        missing = validate_artifacts(artifacts)
        if missing:
            raise SystemExit("Missing release artifacts:\n- " + "\n- ".join(missing))

    if args.mode == "summary":
        graph_input = (
            load_graph_artifact(args.input)
            if args.input_format == "graph"
            else parse_halo_hdf5(args.input)
            if args.input_format == "halo_hdf5"
            else None
        )
        write_graph_summary(graph_input, args.output)
        return 0

    summary = run_pipeline(
        input_format=args.input_format,
        input_path=Path(args.input),
        model_type=args.model_type,
        manifest_path=args.manifest,
        sam_parameters_path=args.sam_parameters,
        sam_id=args.sam_id,
        tree_ids_path=args.tree_ids_file,
        target_indicator_path=args.target_indicator,
        target_indicator_sam_id=args.target_indicator_sam_id,
    )
    write_prediction_rows(summary["rows"], args.output)
    return 0
