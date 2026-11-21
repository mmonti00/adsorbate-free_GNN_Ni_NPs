# GNN models for OH* + H* adsorption on Ni nanoclusters and periodic surfaces

This repository contains the data, trained model checkpoints, training and
inference code, external test systems, and analysis workflows accompanying the
paper. Starting from a MACE-relaxed **bare Ni structure**, the workflow
enumerates candidate OH* + H* adsorption-site pairs across the exposed
surface. The trained models then assign to each candidate:

1. a predicted DFT adsorption energy for dissociated water, referenced to
   H₂O(g); and
2. a motif-retention score indicating whether the initial O and H anchor sets
   and their combined adsorption motif are expected to be retained after
   relaxation.

The adsorption-site labels used throughout are top (`T`), bridge (`B`),
threefold hollow (`H`), and fourfold/folded (`F`). A two-letter motif gives the
O site first and the H* site second; for example, `BH` denotes O on a bridge
and H* in a threefold hollow.

## Repository contents

| Path | Contents |
|---|---|
| `datasets/` | Energy and retention datasets in extxyz format, plus the notebook used to construct initial OH* + H* configurations. |
| `ML_relaxed_bare_geom/MACE/` | MACE-MH-1 geometry-optimization examples and the relaxed bare Ni geometries used for training and testing. |
| `models_checkpoints/` | Released M5 and M6 energy checkpoints, the retention checkpoint, and the exact M6 train/validation structures associated with the released M6 model. |
| `training_scripts/` | M5/M6 energy training, retention training, model analyses, screening/inference, and exact paper/checkpoint split manifests. |
| `test_systems/` | Ranked candidates, DFT initial and optimized structures, reference energies, and validation analyses for the five external test systems; it also contains prediction-only results for larger Ni nanoclusters. |
| `baselines/` | Self-contained Ridge and XGBoost datasets, training code, model files, and reference results. |
| `ni_cutoff_sensitivity_scan/` | Sensitivity analysis for the hybrid Ni–O/Ni–H adsorption-site definitions. |
| `QuantumESPRESSO/` | Representative Quantum ESPRESSO input files showing the DFT settings used in the work. |

Here and in the code, `NP` denotes a finite Ni nanocluster/nanoparticle and
`SLAB` denotes a periodic surface.

## Installation

Python 3.10 or newer is recommended. A basic environment can be created from
the repository root with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and PyTorch Geometric may require platform- or CUDA-specific wheels;
follow their official installation instructions if the generic installation
does not match the local hardware. CPU execution is supported by the GNN
scripts but is substantially slower than GPU execution.

Two external components are not installed by this repository:

- **MACE-MH-1** is required only to regenerate the relaxed bare Ni structures.
  Download the public model and set `model_path` in
  `ML_relaxed_bare_geom/MACE/geo_opt_MACE_NPs.py` or
  `geo_opt_MACE_SLABs.py`. The supplied relaxed bare structures allow the GNN
  workflows to be used without rerunning these optimizations.
- **Quantum ESPRESSO** is required to rerun the DFT calculations. The files in
  `QuantumESPRESSO/` are calculation inputs, not a bundled QE installation.
  Supply compatible pseudopotentials under the directory specified by
  `pseudo_dir` and run them with an appropriate QE executable and computing
  environment.

The Python requirements are intentionally not hardware-pinned. Exact bitwise
agreement after retraining is not guaranteed across PyTorch/PyG, CUDA, or
hardware versions; the released checkpoints and reference outputs are supplied
for direct comparison.

## Released datasets and fixed splits

### Adsorption-energy models

| Dataset | Configurations |
|---|---:|
| M5 nanoclusters | 1,752 |
| M6 nanoclusters | 1,865 |
| M6 periodic surfaces | 216 |
| M6 total | 2,081 |

The exact M6 split used for the released paper checkpoint is:

| Domain | Training | Validation | Total |
|---|---:|---:|---:|
| Nanoclusters | 1,490 | 375 | 1,865 |
| Periodic surfaces | 169 | 47 | 216 |
| **All** | **1,659** | **422** | **2,081** |

This membership is stored in
`training_scripts/paper_splits/M6_energy_model_paper_split.json`. The matching
structures are also distributed as
`models_checkpoints/M6_energy_model_all_train_data.extxyz` and
`models_checkpoints/M6_energy_model_all_valid_data.extxyz`.

### Retention model

The raw paired initial/relaxed datasets contain 5,670 nanocluster pairs and 269
slab pairs. After the validity filters used by the model, the released
checkpoint split contains:

| Domain | Training | Validation | Valid total |
|---|---:|---:|---:|
| Nanoclusters | 4,504 | 1,154 | 5,658 |
| Periodic surfaces | 203 | 65 | 268 |
| **All** | **4,707** | **1,219** | **5,926** |

The exact membership is stored in
`training_scripts/paper_splits/retention_checkpoint_split.json`.

## Reproducing model training

The training files use paths relative to `training_scripts/`, so run them from
that directory:

```bash
cd training_scripts
```

### M5 energy model

```bash
python train_M5_energy_model.py
```

This trains the nanoparticle-only M5 model from scratch.

### M6 energy model

For a new seeded split, leave `FIXED_SPLIT_JSON = None`. To reproduce the exact
train/validation membership used in the paper, set the following near the top
of `train_M6_energy_model.py`:

```python
FIXED_SPLIT_JSON = "paper_splits/M6_energy_model_paper_split.json"
```

Then run:

```bash
python train_M6_energy_model.py
```

By default, M6 is fine-tuned from the released M5 checkpoint. Nanocluster and
surface samples have regression weights of 1 and 4, respectively. The script
also supports training the same architecture from scratch by changing
`FINE_TUNE`.

### Retention model

To reproduce the exact membership associated with the released retention
checkpoint, set these values near the top of
`train_retention_classifier.py`:

```python
FIXED_SPLIT_FILE = "paper_splits/retention_checkpoint_split.json"
USE_FIXED_SPLIT = True
```

Then run:

```bash
python train_retention_classifier.py
```

With `USE_FIXED_SPLIT = False`, the script instead creates a new grouped split
using the configured seed. The training scripts save new checkpoints and
statistics in `training_scripts/`; they do not overwrite the released files in
`models_checkpoints/` unless paths are changed explicitly.

The corresponding post-training analyses can be run, from the same directory,
with:

```bash
python analysis_train_M5_energy_model.py
python analysis_train_M6_energy_model.py
python analysis_train_retention_model.py
```

These analysis scripts expect the outputs produced by the matching training
run in the current directory.

## Screening adsorption-site pairs

`training_scripts/prediction.py` enumerates top, bridge, threefold-hollow, and
fourfold-hollow O/H site pairs on a relaxed bare Ni structure, predicts their
adsorption energies, optionally predicts exact-site retention, and writes a
ranked CSV.

Before running it, review the **USER SETTINGS** block. At minimum, set:

- `SYSTEM_NAME` and the bare-structure path;
- the output CSV path, if the system is not already represented in
  `test_systems/`;
- `MIN_SITE_DIST` and `MAX_SITE_DIST` for the system being screened;
- `USE_PBC` and `TOP_SURFACE_ONLY` for the intended finite or periodic
  geometry;
- `SCORE_NEAR_ONLY`, depending on whether only near O/H site pairs or all pairs
  should be scored; and
- `USE_RETENTION_MODEL`, depending on whether retention scores are
  required.

These geometric settings are deliberately not inferred automatically. For the
surface calculations in the paper, the accepted O/H-site separation window was
selected for the specific system (approximately 1.4–5.5 Å); use the values
appropriate to the geometry being reproduced. The domain used by the GNN is
determined from the structure's PBC flags.

From the repository root, run:

```bash
python training_scripts/prediction.py
```

The ranked CSV files contain the candidate predictions. Each new run also
writes a `*_settings.json` sidecar beside its CSV, recording the checkpoint
hashes, graph cutoffs, candidate-generation options, geometric filters, and
retention thresholds used for that run. Predictions screen candidate pairs on
a bare Ni geometry; they do not replace structural relaxation or DFT
validation.

## Constructing initial OH* + H* geometries

Open `datasets/build_initial_configurations.ipynb` in JupyterLab. It supports:

- `enumerate` mode, which generates allowed O/H site pairs on a complete
  surface or around a selected local environment; and
- `ranking` mode, which constructs selected candidates from a ranked CSV for
  subsequent DFT validation.

The notebook and `prediction.py` use the same site-catalog construction. Review
the settings in the first notebook section before execution.

## External test systems and paper analyses

The five DFT-validated external systems are Ni55, Ni91, Ni130,
Ni(111)-4x3, and Ni(100)-4x4. Their folders contain the ranked predictions,
initial and DFT-optimized structures, and reference energies.

From the repository root, reproduce the energy/ranking and retention analyses
with:

```bash
python test_systems/analysis_energy_test_systems.py
python test_systems/analysis_retention_test_systems.py
```

The first command writes `absolute_relative_energies.pdf` and `rankings.pdf` in
`test_systems/`; the second writes `retention_confusion_matrix.pdf`.

`test_systems/NPs_prediction_only/` contains prediction-only ranked candidates
for larger Ni nanoclusters. The notebook
`test_systems/structural_analysis.ipynb` reproduces the structural and
generalized-coordination-number analyses based on the ranked candidates. Its
introductory cell lists the output associated with each section.

### Training-environment coverage

The local-environment and learned-representation OOD analysis can be reproduced
with:

```bash
python test_systems/environment_ood_analysis/analyze_test_training_environment_ood.py
```

It uses the released M6 checkpoint, its exact train/validation structures, the
five external test systems, and the corresponding bare Ni geometries. Details
and definitions are provided in
`test_systems/environment_ood_analysis/README.md`.

### Ni-contact cutoff sensitivity

From the repository root, run:

```bash
python ni_cutoff_sensitivity_scan/cutoff_scan.py
python ni_cutoff_sensitivity_scan/ni_cutoff_extended.py
```

The first script scans the standard Ni–O/Ni–H cutoffs; the second scans the
extended fourfold cutoffs. Both use the M6 datasets and write into
`ni_cutoff_sensitivity_scan/`. See `ni_cutoff_sensitivity_scan/REPORT.md` for
the protocol and reference results.

## Descriptor-based baselines

The baseline directory is self-contained. It contains the exact M6 training
and validation split represented as structural descriptors, the saved M6
predictions, and the Ridge/XGBoost reference models and results.

```bash
cd baselines
python train_descriptor_baselines.py
```

New outputs are written to `baselines/reproduced_results/`; the results used in
the paper remain under `baselines/reference_results/`. See
`baselines/README.md` for descriptor definitions, selected hyperparameters, and
reference metrics.

## Important conventions

- Adsorption energies are reported in eV and are referenced to H2O(g).
- Standard relaxed-site contact cutoffs are 2.20 Å for Ni–O and 2.00 Å for
  Ni–H. Fourfold assignments use extended cutoffs of 2.30 and 2.20 Å,
  respectively, together with the hybrid contact-count rule implemented in the
  code.
- The external optimized-structure analyses use a 1.90 Å short Ni–H convention; 
  the rationale is documented in the test-analysis scripts and the environment-OOD README.
- Periodic distances are evaluated with minimum-image conventions, and slab
  GNNs include the required periodic images in the local graph.
- Configuration indices in the fixed-split manifests refer to the order of
  frames in the corresponding extxyz files. Do not reorder those files before
  using the manifests.

## Citation

If you use these data, models, or scripts, please cite the accompanying paper.

|   |              [https://doi.org/10.48550/arXiv.2611.20706](https://doi.org/10.48550/arXiv.2611.20706) |
| - | -------------------------------------------------------------------------------------------------- |

## License

The code and materials are released under the MIT License; see `LICENSE`.
