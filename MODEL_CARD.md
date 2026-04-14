# Model Card

## Released Artifacts

- Single model: `models/single/selected_single_model_baseline.pt`
- Ensemble: `models/ensemble/full_48/` with `models/ensemble/full_48_ckpts.txt`
- Shared scaler: `scalers/ensemble_train_6480_scalers.pt`

## Targets

The released predictors output:

- `stellar_mass`
- `sdss_z_band_luminosity`
- `angular_momentum`
- `gas_metal_mass`
- `specific_star_formation_rate`

They also emit:

- `quench_probability`
- `gas_floor_probability`
- `predicted_quenched`
- `predicted_gas_floor`

## Supported Inputs

- halo HDF5 files with the released `forestHalos/...` schema
- graph artifacts with `x`, `edge_index`, and a node identifier

A 17-dimensional SAM parameter vector is required for inference.

## Output Meaning

One output row corresponds to one requested halo target at one supervised redshift layer.

The released model predicts conditional galaxy properties for requested rows. It does not include a learned halo-occupation head and does not infer galaxy existence.

For the mixed discrete-continuous targets:

- `gas_metal_mass` uses a gas-floor state at `-5`
- `specific_star_formation_rate` uses a quenched threshold at `-2`

## Limits

- Generic public inference is not identical to the local paper eval protocol unless a matching target-key source is supplied.
- The release package ships a concat-conditioned inference model only.
- The packaged `full_48` ensemble exposes its first member as `tree_chunk_01__SAM_group_01.pt`; that checkpoint comes from the baseline run and fills the released `(tree_chunk_01, SAM_group_01)` slot.
