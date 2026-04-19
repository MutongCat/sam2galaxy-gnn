# sam2galaxy-gnn

`sam2galaxy-gnn` is the public inference package for the released SAM2Galaxy GNN checkpoints.

## Installation

Use Python 3.9 or newer. The current public package has been tested with `torch 2.0.x`, `torch-geometric 2.6.x`, `numpy 1.26.x`, `h5py 3.12.x`, and `PyYAML 6.x`.

Before installing the package itself, first upgrade the packaging toolchain inside the environment:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Then install a compatible PyTorch stack for your platform. The examples below assume that `torch` and `torch-geometric` are installed before `pip install -e .`.

### Conda

```bash
conda create -n sam2galaxy-gnn python=3.9
conda activate sam2galaxy-gnn
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install "torch>=2.0,<2.1"
python -m pip install "torch-geometric>=2.6,<2.7"
python -m pip install -e .
```

### venv

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install "torch>=2.0,<2.1"
python -m pip install "torch-geometric>=2.6,<2.7"
python -m pip install -e .
```

Install from the repository root:

```bash
cd sam2galaxy-gnn
python -m pip install -e .
```

If you already have a compatible PyTorch install, `pip install -e .` installs the remaining package dependencies. After installation, the CLI command `sam2galaxy-gnn` is available from the environment.

If `h5py` fails to build because the system HDF5 library is unavailable, install `h5py` and `hdf5` through your package manager or through `conda-forge` before installing this package.

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

The large halo HDF5 input is distributed separately from the Git repository as a compressed release asset. Download `galacticus_Uchuu_rockstar.hdf5.gz`, decompress it, and place the extracted file at `data/halo_hdf5/galacticus_Uchuu_rockstar.hdf5`. See [data/halo_hdf5/README.md](data/halo_hdf5/README.md).

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

1. Validate that the packaged models, scaler, mapping, and example assets are present:

```bash
python scripts/validate_release.py
```

2. Run the recommended first example on the packaged graph input:

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

This writes a prediction catalog to `/tmp/example_single_from_graph.csv`.

3. If you want to run from the original halo HDF5 instead of the packaged graph, first download `galacticus_Uchuu_rockstar.hdf5.gz`, run `gunzip galacticus_Uchuu_rockstar.hdf5.gz`, place the extracted file at `data/halo_hdf5/galacticus_Uchuu_rockstar.hdf5`, and then run:

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

To regenerate the packaged example outputs and summaries, use:

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
