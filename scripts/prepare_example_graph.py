#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam2galaxy_gnn.graph_builder import build_graph_from_halo_hdf5, write_graph_summary
from sam2galaxy_gnn.runtime import load_sam_parameters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-halo-hdf5", required=True)
    parser.add_argument("--output-graph", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--sam-parameters", default=None)
    parser.add_argument("--sam-id", type=int, default=None)
    parser.add_argument("--tree-ids-file", required=True)
    parser.add_argument("--node-index-to-tree-id", required=True)
    args = parser.parse_args()

    sam_params = load_sam_parameters(args.sam_parameters, sam_id=args.sam_id) if args.sam_parameters else None
    graph_input = build_graph_from_halo_hdf5(
        args.input_halo_hdf5,
        sam_parameters=sam_params,
        tree_ids_path=args.tree_ids_file,
        node_index_to_tree_id_path=args.node_index_to_tree_id,
    )
    torch.save(graph_input.payload, args.output_graph)
    output = write_graph_summary(graph_input, Path(args.output_summary))
    print(args.output_graph)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
