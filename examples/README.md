# Examples

The example assets are organized into one shared `input/` area and one shared `output/` area.

## `input/tree_input`

These files are for the `halo_hdf5 -> graph -> predict` route:

- `tree_ids.txt`
- `sam_parameters.dat`
- `target_indicator.csv`
- `truth_values.csv`

Use `tree_ids.txt` together with the packaged halo HDF5 input to select the example tree subset.

## `input/graph_input`

These files are for the direct `graph -> predict` route:

- `graph.pt`
- `graph_summary.json`
- `sam_parameters.dat`
- `target_indicator.csv`
- `truth_values.csv`

The graph and tree inputs describe the same example set:

- `8` representative test SAMs
- `1024` trees from the test tree chunk defined in the paper

`sam_parameters.dat` is the example SAM table. It contains one row per `sam_id` plus the 17 SAM parameters. Use `--sam-id` to select one row.

`target_indicator.csv` defines which halo rows should be predicted. It is a row-key file, not a learned halo occupation product. Its key columns are:

- `sam_id`
- `node_index`
- `tree_id`
- `z_index`
- `redshift`

`truth_values.csv` contains the same key columns plus the five target properties and the two truth-state labels:

- `quench_truth`
- `gas_floor_truth`

## `output`

Example outputs are split by model family:

- `output/single/`
- `output/ensemble/`

Each directory contains one prediction catalog per `sam_id`:

- `sam_XXXX_predictions.csv`

The example also includes two aggregate summaries:

- `output/single_summary.json`
- `output/ensemble_summary.json`

Those summaries are computed across all example SAMs and trees.

## Metric Semantics

For `gas_metal_mass` and `specific_star_formation_rate`, the regression metrics are computed only on rows where both truth and prediction remain in the continuous regime.

For the ensemble outputs, `gas_metal_mass` and `specific_star_formation_rate` are not aggregated by a plain mean over final predictions. The release package follows the aggregation rule:

- first determine the hard floor / quenched state for each ensemble member
- use a majority vote on that hard state
- if the majority is continuous, average only the continuous members
- if the majority is floor / quenched, emit the floor / quenched value
