#!/usr/bin/env python3
"""Predict the external test configurations with M6-MACE trained from scratch.

The from-scratch model (``../models_checkpoints/M6_energy_model_from_scratch.pt``)
is the SI control obtained with ``training_scripts/train_M6_energy_model.py`` and
``FINE_TUNE = False`` on the paper split. For every DFT-evaluated configuration of
the five external test systems, the graph of the final (DFT-relaxed) O/H anchor
sets is built with ``training_scripts/prediction.py`` on the MACE-relaxed bare
structure, exactly as for the screening rankings, and the adsorption energy is
predicted.

``make_baseline_figures.py`` reads the resulting CSV. The prediction is kept in a
separate script because PyTorch and XGBoost can load conflicting OpenMP runtimes
when imported into the same process.

Run from this directory: ``python predict_external_M6_from_scratch.py``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from ase.io import read

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
TRAINING_DIR = REPOSITORY_ROOT / "training_scripts"
TEST_SYSTEMS_DIR = REPOSITORY_ROOT / "test_systems"
MACE_BARE_DIR = REPOSITORY_ROOT / "ML_relaxed_bare_geom" / "MACE" / "testing"
CHECKPOINT = REPOSITORY_ROOT / "models_checkpoints" / "M6_energy_model_from_scratch.pt"
OUTPUT = HERE / "reproduced_results" / "M6_from_scratch_external_predictions.csv"
BARE_FILES = {
    "Ni55": "Ni55_MACE_opt.xyz",
    "Ni91": "Ni91_MACE_opt.xyz",
    "Ni130": "Ni130_MACE_opt.xyz",
    "Ni111_4x3": "Ni111_4x3_MACE_opt.extxyz",
    "Ni100_4x4": "Ni100_4x4_MACE_opt.extxyz",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()

    # prediction.py and the energy training module resolve paths from their own directory.
    os.chdir(TRAINING_DIR)
    sys.path.insert(0, str(TRAINING_DIR))
    sys.path.insert(1, str(TEST_SYSTEMS_DIR))
    import prediction
    import analysis_energy_test_systems as validation

    model = prediction.load_model(str(checkpoint))[0]
    rows = []
    for spec in validation.SYSTEMS:
        frame, _ = validation.process_system(TEST_SYSTEMS_DIR, spec)
        frame = frame.dropna(subset=["DFT_Eads_eV", "M6_Eads_eV"])
        bare = read(MACE_BARE_DIR / BARE_FILES[spec.folder])
        prediction.CURRENT_ATOMS = bare
        domain = prediction.domain_label_for(bare)
        for _, row in frame.iterrows():
            o_anchors = [int(i) for i in str(row["final_o_anchor_idx"]).split(",")]
            h_anchors = [int(i) for i in str(row["final_h_anchor_idx"]).split(",")]
            data, meta = prediction.build_candidate_data(
                bare.get_positions(), o_anchors, h_anchors, domain_label=domain
            )
            if meta.get("pbc_image_ambiguous", False):
                raise ValueError(f"{spec.label} config {row['config']}: ambiguous periodic image")
            rows.append({
                "system": spec.label,
                "config": int(row["config"]),
                "DFT_Eads_eV": float(row["DFT_Eads_eV"]),
                "pred_Eads_eV": prediction.predict_candidate(model, data),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
