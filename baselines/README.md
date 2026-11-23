# Descriptor-based baselines

This directory contains the processed dataset and training code used for the Ridge and XGBoost comparisons reported in Section S5 of the Supporting Information, together with the predictions of M6-MACE and of the same GNN trained from scratch on the same split.

## Training and validation data

`datasets/baseline_dataset.csv` is the exact table supplied to the baseline models. It contains:

- 1,659 training configurations: 1,490 nanoclusters and 169 periodic surfaces;
- 422 validation configurations: 375 nanoclusters and 47 periodic surfaces.

Unlike M6-MACE, which was obtained by fine-tuning the M5 checkpoint, the descriptor-based baselines were trained from scratch on the exact M6 training split. The original M6 validation split was used for hyperparameter selection and evaluation. To separate the contribution of the graph representation from that of the training strategy, the M6-MACE architecture was also trained from scratch on the same split, without initialization from M5 (M6-MACE (from scratch); checkpoint `../models_checkpoints/M6_energy_model_from_scratch.pt`). It is reproduced by running `../training_scripts/train_M6_energy_model.py` with `FINE_TUNE = False`; training ends by early stopping after 267 epochs.

The target is the DFT adsorption energy. The structural descriptors are evaluated on the corresponding MACE-relaxed bare Ni geometry using the final adsorption motif and O/H anchor sets assigned from the DFT-relaxed adsorbate structure.

Important columns include:

- `split`: `train` or `validation`;
- `domain`: nanocluster (`NP`) or periodic surface (`SLAB`);
- `motif`: combined O/H adsorption-site motif;
- `GCN_O`, `GCN_H`: generalized coordination numbers of the O and H binding sites;
- `d_OH_A`: distance between the two adsorption-site centroids in angstrom;
- `Eads_DFT_eV`: DFT adsorption-energy target in eV;
- `sample_weight`: 1 for nanoclusters and 4 for periodic surfaces.

`datasets/baseline_dataset_audit.json` records the source counts, split composition and descriptor protocol. `datasets/M6_energy_model_predictions.csv` and `datasets/M6_from_scratch_predictions.csv` contain the saved predictions of M6-MACE and M6-MACE (from scratch) used for the side-by-side validation metrics.

## Reproducing the models

Install NumPy, pandas, SciPy, scikit-learn, XGBoost, ASE, matplotlib and joblib. All input paths are relative to the repository, and every script writes new outputs to `reproduced_results/`. From this directory, run:

```bash
python build_baseline_dataset.py             # optional: rebuild the descriptor table
python train_descriptor_baselines.py         # hyperparameter search, fit, validation metrics
python predict_external_M6_from_scratch.py   # optional: recompute the from-scratch external predictions
python make_baseline_figures.py              # Figures S8 and S9, external-test evaluation
```

- `build_baseline_dataset.py` recomputes `datasets/baseline_dataset.csv` from the released M6 train/validation structures in `../models_checkpoints/` and the MACE-relaxed bare geometries in `../ML_relaxed_bare_geom/MACE/training/`. Slab descriptors are evaluated with `periodic_site.py`: the anchors of a site are placed in one compact set of periodic images, GCN neighbours are counted per periodic image relative to those anchors, and the anchors themselves are excluded.
- `train_descriptor_baselines.py` performs the Ridge and XGBoost hyperparameter searches, evaluates the selected models on the fixed validation set, and includes the supplied M6-MACE predictions in the comparison metrics. Use `--dataset reproduced_results/baseline_dataset.csv` to train on a rebuilt table.
- `predict_external_M6_from_scratch.py` predicts the 199 DFT-evaluated external configurations with M6-MACE (from scratch), building each graph with `../training_scripts/prediction.py` at the final (DFT-relaxed) anchor sets, as for the screening rankings. It is a separate script because PyTorch and XGBoost can load conflicting OpenMP runtimes in one process.
- `make_baseline_figures.py` draws the internal-validation parity grid for Ridge, XGBoost, M6-MACE (from scratch) and M6-MACE (Figure S8), and evaluates the four models on the five external test systems (Figure S9). The external configurations, final motifs and anchor sets are parsed with `../test_systems/analysis_energy_test_systems.py`, and the descriptors are computed on the MACE-relaxed bare test structures. By default it uses the models and the from-scratch external predictions in `reference_results/`; pass `--results-dir reproduced_results` to use newly trained baselines (copy `M6_from_scratch_external_predictions.csv` there, or regenerate it with the script above). It also writes `internal_validation_metrics.csv` with the internal-validation errors of all four models.

The `reference_results/` directory contains the metrics, predictions, selected hyperparameters, portable model files and external-test results used for the paper.

The portable JSON model files are included instead of Python joblib files because JSON is less sensitive to package and Python versions.

## Reference internal-validation results

| Model | Configurations | Absolute MAE (eV) | Absolute RMSE (eV) | Relative MAE (eV) | Relative RMSE (eV) |
|---|---|---:|---:|---:|---:|
| Ridge | All | 0.111 | 0.142 | 0.320 | 0.375 |
| Ridge | Nanoclusters | 0.117 | 0.147 | 0.352 | 0.396 |
| Ridge | Surfaces | 0.064 | 0.090 | 0.065 | 0.096 |
| XGBoost | All | 0.072 | 0.097 | 0.143 | 0.186 |
| XGBoost | Nanoclusters | 0.077 | 0.102 | 0.158 | 0.197 |
| XGBoost | Surfaces | 0.030 | 0.042 | 0.025 | 0.038 |
| M6-MACE (from scratch) | All | 0.034 | 0.051 | 0.045 | 0.060 |
| M6-MACE (from scratch) | Nanoclusters | 0.033 | 0.050 | 0.046 | 0.061 |
| M6-MACE (from scratch) | Surfaces | 0.041 | 0.060 | 0.032 | 0.055 |
| M6-MACE | All | 0.032 | 0.049 | 0.045 | 0.060 |
| M6-MACE | Nanoclusters | 0.031 | 0.047 | 0.046 | 0.060 |
| M6-MACE | Surfaces | 0.045 | 0.064 | 0.035 | 0.060 |

Relative energies are referenced, within each system, to the configuration with the lowest DFT adsorption energy; the same configuration is used as the reference for the predicted energies. The unrounded values are reported in `reference_results/baseline_metrics.csv` (Ridge, XGBoost and M6-MACE) and `reference_results/internal_validation_metrics.csv` (all four models).

## Reference external-test results

| System | Model | Configurations | Absolute MAE (eV) | Relative MAE (eV) |
|---|---|---:|---:|---:|
| Ni55 | Ridge | 49 | 0.232 | 0.136 |
| Ni55 | XGBoost | 49 | 0.310 | 0.167 |
| Ni55 | M6-MACE (from scratch) | 49 | 0.183 | 0.065 |
| Ni55 | M6-MACE | 49 | 0.187 | 0.043 |
| Ni91 | Ridge | 44 | 0.162 | 0.371 |
| Ni91 | XGBoost | 44 | 0.219 | 0.430 |
| Ni91 | M6-MACE (from scratch) | 44 | 0.098 | 0.297 |
| Ni91 | M6-MACE | 44 | 0.109 | 0.114 |
| Ni130 | Ridge | 35 | 0.124 | 0.150 |
| Ni130 | XGBoost | 35 | 0.157 | 0.153 |
| Ni130 | M6-MACE (from scratch) | 35 | 0.088 | 0.151 |
| Ni130 | M6-MACE | 35 | 0.074 | 0.075 |
| Ni(111)-4x3 | Ridge | 26 | 0.029 | 0.047 |
| Ni(111)-4x3 | XGBoost | 26 | 0.027 | 0.046 |
| Ni(111)-4x3 | M6-MACE (from scratch) | 26 | 0.041 | 0.027 |
| Ni(111)-4x3 | M6-MACE | 26 | 0.022 | 0.022 |
| Ni(100)-4x4 | Ridge | 45 | 0.055 | 0.045 |
| Ni(100)-4x4 | XGBoost | 45 | 0.074 | 0.059 |
| Ni(100)-4x4 | M6-MACE (from scratch) | 45 | 0.075 | 0.057 |
| Ni(100)-4x4 | M6-MACE | 45 | 0.079 | 0.050 |

Unrounded values, RMSEs and R² are reported in `reference_results/baseline_external_metrics.csv`; the per-configuration predictions are in `reference_results/baseline_external_predictions.csv`, and those of M6-MACE (from scratch) also in `reference_results/M6_from_scratch_external_predictions.csv`.
