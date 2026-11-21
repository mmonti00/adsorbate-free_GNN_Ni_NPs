# Descriptor-based baselines

This directory contains the processed dataset and training code used for the Ridge and XGBoost comparisons reported in Section S5 of the Supporting Information.

## Training and validation data

`datasets/baseline_dataset.csv` is the exact table supplied to the baseline models. It contains:

- 1,659 training configurations: 1,490 nanoclusters and 169 periodic surfaces;
- 422 validation configurations: 375 nanoclusters and 47 periodic surfaces.

Unlike M6-MACE, which was obtained by fine-tuning the M5 checkpoint, the descriptor-based baselines were trained from scratch on the exact M6 training split. The original M6 validation split was used for hyperparameter selection and evaluation.

The target is the DFT adsorption energy. The structural descriptors are evaluated on the corresponding MACE-relaxed bare Ni geometry using the final adsorption motif and O/H anchor sets assigned from the DFT-relaxed adsorbate structure.

Important columns include:

- `split`: `train` or `validation`;
- `domain`: nanocluster (`NP`) or periodic surface (`SLAB`);
- `motif`: combined O/H adsorption-site motif;
- `GCN_O`, `GCN_H`: generalized coordination numbers of the O and H binding sites;
- `d_OH_A`: distance between the two adsorption-site centroids in angstrom;
- `Eads_DFT_eV`: DFT adsorption-energy target in eV;
- `sample_weight`: 1 for nanoclusters and 4 for periodic surfaces.

`datasets/baseline_dataset_audit.json` records the source counts, split composition and descriptor protocol. `datasets/M6_energy_model_predictions.csv` contains the saved M6-MACE predictions used for the side-by-side validation metrics.

## Reproducing the models

Install NumPy, pandas, SciPy, scikit-learn, XGBoost, matplotlib and joblib. From this directory, run:

```bash
python train_descriptor_baselines.py
```

All input paths are relative to this directory. The script performs the Ridge and XGBoost hyperparameter searches, evaluates the selected models on the fixed validation set, includes the supplied M6-MACE predictions in the comparison metrics, and writes new outputs to `reproduced_results/`. The `reference_results/` directory contains the metrics, predictions, selected hyperparameters and portable model files used for the paper.

The portable JSON model files are included instead of Python joblib files because JSON is less sensitive to package and Python versions.

## Reference internal-validation results

| Model | All configurations MAE (eV) | Nanoclusters MAE (eV) | Surfaces MAE (eV) |
|---|---:|---:|---:|
| Ridge | 0.114 | 0.119 | 0.077 |
| XGBoost | 0.073 | 0.077 | 0.043 |
| M6-MACE | 0.031 | 0.030 | 0.040 |

The unrounded values and relative-energy metrics are reported in `reference_results/baseline_metrics.csv`.
