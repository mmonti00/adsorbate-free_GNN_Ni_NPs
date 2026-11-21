# Training-environment coverage of the optimized DFT test systems

This directory contains an analysis of how closely the local adsorption
environments in the optimized DFT test configurations resemble environments
seen during M6 training.

The analysis is intended as a diagnostic of structural interpolation and
extrapolation. It is not a formal probabilistic uncertainty estimate.

## Files

- `test_training_environment_ood.pdf`: summary figure.
- `test_environment_ood_summary.csv`: one-row-per-system summary.
- `test_environment_ood_details.csv`: GCN classification and consensus OOD
  flags for every optimized test configuration.
- `learned_embedding_ood_details.csv`: distances and thresholds for every
  representation/metric combination.
- `learned_embedding_metric_summary.csv`: representation-specific summary.
- `analyze_test_training_environment_ood.py`: reproducible analysis script.

## Data used

The reference population is the exact split distributed with the released M6
checkpoint:

- 1,659 training configurations;
- 422 validation configurations.

The released training split contains one documented legacy configuration,
`Ni45_conf1_0078`, whose geometry is O*+H*+H* rather than OH*+H*.  It is kept
only to reproduce the released checkpoint and is processed with the historical
closest-H assignment.  The default new-split training workflow rejects this
frame.  The analysis script activates this compatibility rule explicitly so
that the reference indices remain aligned with the checkpoint split.

The external set contains 199 optimized DFT configurations:

| System | Configurations |
|---|---:|
| Ni55 | 49 |
| Ni91 | 44 |
| Ni130 | 35 |
| Ni(111)-4x3 | 26 |
| Ni(100)-4x4 | 45 |

For both training and test configurations, the O and H anchor atoms are
identified from the relaxed adsorbate-containing geometry. The corresponding
anchor indices are then mapped onto the MACE-relaxed bare Ni structure used to
construct the M6 graph. Thus, the comparison uses the same bare structural
representation that the model receives during training and inference.

Periodic cells retain their lattice and PBC metadata. Adsorbate-anchor
distances, Ni-Ni connectivity, graph construction, and GCN calculations use
periodic images or minimum-image distances where appropriate.

## Diagnostic 1: motif-resolved GCN coverage

Each test configuration is compared only with training configurations from
the same domain (nanoparticle or slab) and with the same final motif.

The environment is represented by `(GCN_O, GCN_H)`. Site GCN values are
calculated from the union of the Ni neighbours of the anchor atoms, excluding
the anchor atoms themselves, with site-specific normalization:

- T: `CN_max = 12`
- B: `CN_max = 18`
- H: `CN_max = 22`
- F: `CN_max = 26`

The GCN categories in the left panel are:

- **Exact GCN pair**: both GCN values match a training pair within numerical
  tolerance.
- **Near**: the nearest same-motif training configuration has
  `max(|Delta GCN_O|, |Delta GCN_H|) <= 0.15`.
- **Inside marginal ranges**: both descriptors lie inside their respective
  same-motif training ranges, but their combination is not near an observed
  training pair.
- **Outside training range**: at least one descriptor lies outside the
  same-motif training range.
- **Motif unseen**: the motif is absent from the corresponding training
  domain.

“GCN extrapolation” in the right panel combines the last two categories.

## Diagnostic 2: learned-representation distances

Two M6 representations are extracted from the released checkpoint:

- the 384-dimensional pooled vector obtained by concatenating the global,
  O-focused, and H-focused graph pools;
- the 64-dimensional hidden representation immediately before the final
  energy output layer.

For each representation, a test configuration is compared with training
configurations of the same domain and motif using:

1. cosine distance in the original representation;
2. Euclidean distance after standardization and PCA whitening, retaining 95%
   of the training variance.

OOD thresholds are derived from validation-to-training nearest-neighbour
distances. A separate 95th-percentile threshold is used for each domain and
motif when at least ten validation examples are available; otherwise the
domain-level validation threshold is used.

The two reported learned-representation flags are:

- **Energy-head embedding**: above threshold with both metrics in the
  64-dimensional energy-head representation.
- **All-embedding consensus**: above threshold with both metrics in both the
  64-dimensional and 384-dimensional representations.

An unseen same-domain motif is flagged as OOD by construction.

## Results

| System | Inside training GCN ranges | Clear GCN OOD | Energy-head OOD | All-embedding consensus OOD |
|---|---:|---:|---:|---:|
| Ni55 | 48/49 (98.0%) | 1/49 (2.0%) | 6/49 (12.2%) | 6/49 (12.2%) |
| Ni91 | 40/44 (90.9%) | 4/44 (9.1%) | 21/44 (47.7%) | 16/44 (36.4%) |
| Ni130 | 20/35 (57.1%) | 15/35 (42.9%) | 23/35 (65.7%) | 23/35 (65.7%) |
| Ni(111)-4x3 | 24/26 (92.3%) | 2/26 (7.7%) | 3/26 (11.5%) | 3/26 (11.5%) |
| Ni(100)-4x4 | 45/45 (100%) | 0/45 (0%) | 6/45 (13.3%) | 6/45 (13.3%) |

The main conclusions are:

- Ni55 is predominantly interpolative.
- Ni91 retains strong marginal GCN overlap with training, but a substantial
  subset occupies less familiar combinations in the learned representation.
- Ni130 presents the clearest environment-level distribution shift and
  remains distinct even in the task-specific energy-head representation.
- Ni(111)-4x3 is mostly represented. The two HT GCN extrapolations are the
  known structures affected by the 1.90 A Ni-H analysis convention.
- Ni(100)-4x4 has complete scalar GCN coverage. Its six learned-representation
  outliers are all FF configurations (`conf03`, `conf04`, `conf05`, `conf06`,
  `conf17`, and `conf41`), showing that the complete graph distinguishes
  structural information that is not retained by the two scalar GCN values.
- TF is absent from the NP training set and occurs once in Ni91 and once in
  Ni130.

## Reproducing the analysis

From the repository root, run:

```bash
python test_systems/environment_ood_analysis/analyze_test_training_environment_ood.py
```

The script recreates this directory and overwrites the PDF and CSV outputs.
It expects the released M6 checkpoint and its train/validation extxyz files in
`models_checkpoints`, and the MACE-relaxed bare structures in
`ML_relaxed_bare_geom/MACE`.
