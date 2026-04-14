# sam2galaxy-gnn

`sam2galaxy-gnn` is the public inference package for the released SAM2Galaxy GNN checkpoints.

## Installation

Create a Python 3.10+ environment, install PyTorch for your platform, then install this package from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e .
```

If you already have a compatible PyTorch install, `pip install -e .` will install the remaining package dependencies. After installation, the CLI command `sam2galaxy-gnn` is available from the environment.

## What It Does

- loads a halo HDF5 or graph input
- attaches a 17-parameter SAM vector
- predicts five galaxy properties
- supports single-model and ensemble inference

Supported input routes:

1. `halo_hdf5`
   Use the released `forestHalos/...` HDF5 schema and build a graph on the fly.
2. `graph`
   Use a prebuilt graph artifact directly.

The public `halo_hdf5 -> graph` route is inference-only. It does not build labels or train-time cache fields.

The large halo HDF5 input is distributed separately from the Git repository. Put `galacticus_Uchuu_rockstar.hdf5` at `data/halo_hdf5/galacticus_Uchuu_rockstar.hdf5` after downloading it from the project release assets or archival record. See [data/halo_hdf5/README.md](data/halo_hdf5/README.md).

## Output Semantics

One output row corresponds to one requested halo target at one supervised redshift layer.

If no `--target-indicator` is provided, the package predicts all halo nodes on the matched output redshift layers. If `--target-indicator` is provided, it predicts only the rows defined by that file.

The model predicts conditional galaxy properties. It does not decide whether a halo hosts a galaxy.

## Layout

- `src/sam2galaxy_gnn/`: package code
- `scripts/`: command-line entrypoints
- `configs/`: release manifest and inference configs
- `models/`: released checkpoints
- `scalers/`: released scaler
- `data/halo_hdf5/`: packaged halo HDF5 input
- `data/tree_mappings/`: packaged `nodeIndex -> tree_id` mapping
- `examples/input/tree_input/`: example halo-HDF5 inputs
- `examples/input/graph_input/`: example graph inputs
- `examples/output/`: example prediction outputs and summaries

## Quickstart

Validate that the packaged models, scaler, mapping, and example assets are present:

```bash
python scripts/validate_release.py
```

If you plan to use the `halo_hdf5` route, download `galacticus_Uchuu_rockstar.hdf5` first and place it at `data/halo_hdf5/galacticus_Uchuu_rockstar.hdf5`.

Run single-model inference on the example graph input:

```bash
sam2galaxy-gnn \
  --manifest configs/release_manifest.json \
  --input-format graph \
  --input examples/input/graph_input/graph.pt \
  --sam-parameters examples/input/graph_input/sam_parameters.dat \
  --sam-id 269 \
  --target-indicator examples/input/graph_input/target_indicator.csv \
  --output /tmp/example_single_from_graph.csv \
  --mode predict \
  --model-type single
```

Run single-model inference from the halo HDF5 route:

```bash
sam2galaxy-gnn \
  --manifest configs/release_manifest.json \
  --input-format halo_hdf5 \
  --input data/halo_hdf5/galacticus_Uchuu_rockstar.hdf5 \
  --tree-ids-file examples/input/tree_input/tree_ids.txt \
  --sam-parameters examples/input/tree_input/sam_parameters.dat \
  --sam-id 269 \
  --target-indicator examples/input/tree_input/target_indicator.csv \
  --output /tmp/example_single_from_halo_hdf5.csv \
  --mode predict \
  --model-type single
```

Generate the packaged example outputs:

```bash
python scripts/generate_example_eval_outputs.py \
  --manifest configs/release_manifest.json \
  --graph examples/input/graph_input/graph.pt \
  --target-indicator examples/input/graph_input/target_indicator.csv \
  --truth-values examples/input/graph_input/truth_values.csv \
  --sam-parameters examples/input/graph_input/sam_parameters.dat \
  --output-single-dir /tmp/example_output/single \
  --output-ensemble-dir /tmp/example_output/ensemble \
  --output-single-summary /tmp/example_output/single_summary.json \
  --output-ensemble-summary /tmp/example_output/ensemble_summary.json
```

## Notes

- Python package name: `sam2galaxy_gnn`
- Install from the repository root so the packaged example files are available at the documented paths.
- The public predictor requires a 17-parameter SAM vector.
- The example uses 8 test SAMs and 1024 trees.
- `data/tree_mappings/node_index_to_tree_id.pt` is used to select the same trees from the original HDF5 input.
