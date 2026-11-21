#!/usr/bin/env python3
"""Evaluate the trained M5 nanoparticle adsorption-energy model.

The analysis rebuilds the same bare-site graphs used for training, verifies
that the checkpoint preprocessing settings still match the training module,
restores the saved train/validation split, and reports errors overall and by
nanoparticle, motif, and anchor rule. It also writes a PDF parity plot, a full
CSV error table, a table of the largest errors, and annotated extxyz files for
the ten configurations with the largest errors.

The reported O-H* distance is the distance between O and the independently
adsorbed H atom, not the O-H bond within OH*. Run this script from its own
directory so that the relative paths resolve as provided.
"""

import csv
import os
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import write
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from sklearn.metrics import r2_score

from train_M5_energy_model import (
    BareSiteGCNN,
    build_pyg_dataset,
    load_initial_reference_structures,
    ID_TO_MOTIF,
    DEVICE,
    EXTXYZ_FILE,
    MODEL_OUT,
    NI_O_CUTOFF,
    NI_O_FOLDED_CUTOFF,
    O_FOLDED_N_SHORT,
    O_FOLDED_N_TOTAL,
    MAX_STANDARD_O_ANCHORS,
    NI_H_CUTOFF,
    NI_H_FOLDED_CUTOFF,
    H_FOLDED_N_SHORT,
    H_FOLDED_N_TOTAL,
    MAX_STANDARD_H_ANCHORS,
    OH_BOND_CUTOFF,
    LOCAL_CUTOFF,
    EDGE_CUTOFF,
    N_RBF,
    RBF_DMIN,
    RBF_DMAX,
    RBF_GAMMA,
    SITE_TYPES,
)

# ============================================================
# INPUTS AND OUTPUTS
# ============================================================

initial_refs = load_initial_reference_structures()
dataset, metadata, frames, group_labels, _ = build_pyg_dataset(EXTXYZ_FILE, initial_refs)

MODEL_PATH = MODEL_OUT
N_WORST_TABLE = 50
N_WORST_STRUCTURES = 10
O_H_ADS_DISTANCE_THRESHOLD = 2.20

OUTDIR = "check_train_M5_energy_model"
PARITY_FILE = os.path.join(OUTDIR, "parity_train_validation.pdf")
WORST_STRUCTURES_DIR = os.path.join(OUTDIR, "worst_10_structures")
WORST_TABLE_FILE = os.path.join(OUTDIR, "worst_50_configurations.csv")
ERROR_TABLE_FILE = os.path.join(OUTDIR, "model_errors.csv")

os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})


def compute_o_h_ads_distance(atoms):
    """Return the distance between O and the independently adsorbed H*."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    o_indices = [i for i, s in enumerate(symbols) if s == "O"]
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]

    if len(o_indices) != 1 or len(h_indices) != 2:
        return None

    o_idx = o_indices[0]
    o_pos = positions[o_idx]

    # The closest H belongs to OH*; the other is H*.
    dists = [(h_idx, np.linalg.norm(positions[h_idx] - o_pos)) for h_idx in h_indices]
    dists = sorted(dists, key=lambda x: x[1])

    h_ads_idx = dists[1][0]

    o_h_ads_distance = np.linalg.norm(positions[o_idx] - positions[h_ads_idx])
    return float(o_h_ads_distance)


# ============================================================
# CHECKPOINT, MODEL, AND SAVED SPLIT
# ============================================================

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

settings = ckpt.get("settings", {})
expected_settings = {
    "OH_BOND_CUTOFF": OH_BOND_CUTOFF,
    "NI_O_CUTOFF": NI_O_CUTOFF,
    "NI_O_FOLDED_CUTOFF": NI_O_FOLDED_CUTOFF,
    "O_FOLDED_N_SHORT": O_FOLDED_N_SHORT,
    "O_FOLDED_N_TOTAL": O_FOLDED_N_TOTAL,
    "MAX_STANDARD_O_ANCHORS": MAX_STANDARD_O_ANCHORS,
    "NI_H_CUTOFF": NI_H_CUTOFF,
    "NI_H_FOLDED_CUTOFF": NI_H_FOLDED_CUTOFF,
    "H_FOLDED_N_SHORT": H_FOLDED_N_SHORT,
    "H_FOLDED_N_TOTAL": H_FOLDED_N_TOTAL,
    "MAX_STANDARD_H_ANCHORS": MAX_STANDARD_H_ANCHORS,
    "LOCAL_CUTOFF": LOCAL_CUTOFF,
    "EDGE_CUTOFF": EDGE_CUTOFF,
    "N_RBF": N_RBF,
    "RBF_DMIN": RBF_DMIN,
    "RBF_DMAX": RBF_DMAX,
    "RBF_GAMMA": RBF_GAMMA,
}

print("\n=== CHECKPOINT PREPROCESSING CHECK ===")
for key, expected in expected_settings.items():
    if key not in settings:
        raise RuntimeError(f"Checkpoint is missing preprocessing setting {key!r}.")
    actual = settings[key]
    if float(actual) != float(expected):
        raise RuntimeError(
            f"Checkpoint/module mismatch for {key}: "
            f"checkpoint={actual}, imported module={expected}."
        )
    print(f"{key} = {actual}")

if list(settings.get("SITE_TYPES", [])) != list(SITE_TYPES):
    raise RuntimeError(
        "Checkpoint/module mismatch for SITE_TYPES: "
        f"checkpoint={settings.get('SITE_TYPES')}, imported module={list(SITE_TYPES)}."
    )

if not bool(ckpt.get("energy_only", False)):
    raise RuntimeError("The checkpoint is not marked as energy-only.")

train_idx = set(ckpt["train_idx"])
val_idx = set(ckpt["val_idx"])
overlap = train_idx & val_idx
if overlap:
    raise RuntimeError(
        f"Checkpoint split contains {len(overlap)} overlapping indices."
    )

expected_idx = set(range(len(dataset)))
saved_idx = train_idx | val_idx
if saved_idx != expected_idx:
    raise RuntimeError(
        "Checkpoint split does not match the rebuilt dataset: "
        f"{len(expected_idx - saved_idx)} missing and "
        f"{len(saved_idx - expected_idx)} out-of-range indices."
    )

model = BareSiteGCNN(
    node_dim=ckpt["node_dim"],
    edge_dim=ckpt["edge_dim"],
    hidden_dim=ckpt["hidden_dim"],
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()
id_to_motif = ckpt.get("id_to_motif", ID_TO_MOTIF)


# ============================================================
# RUN INFERENCE
# ============================================================

y_true_energy = []
y_pred_energy = []
y_true_motif = []

errors = []

with torch.no_grad():
    for i, (data, meta) in enumerate(zip(dataset, metadata)):
        data = data.to(DEVICE)

        batch_vec = torch.zeros(data.x.shape[0], dtype=torch.long, device=DEVICE)

        e_pred = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch_vec,
        )

        e_pred = float(e_pred.item())
        e_true = float(data.y_energy.item())
        m_true = int(data.y_motif.item())

        global_idx = i
        frame_idx = int(meta["frame_idx"])

        o_h_ads_distance = compute_o_h_ads_distance(frames[global_idx])

        if global_idx in train_idx:
            split = "train"
        elif global_idx in val_idx:
            split = "validation"
        else:
            split = "unknown"

        y_true_energy.append(e_true)
        y_pred_energy.append(e_pred)
        y_true_motif.append(m_true)

        errors.append({
            "global_idx": global_idx,
            "frame_idx": frame_idx,
            "true_energy": e_true,
            "pred_energy": e_pred,
            "error": e_pred - e_true,
            "abs_error": abs(e_pred - e_true),
            "true_motif": id_to_motif[m_true],
            "group": group_labels[i],
            "o_h_ads_distance": o_h_ads_distance,
            "o_anchor_rule": meta.get("o_anchor_rule", "unknown"),
            "o_short_contact_count": meta.get("o_short_contact_count", np.nan),
            "o_extended_contact_count": meta.get("o_extended_contact_count", np.nan),
            "o_anchor_idx": meta.get("o_anchor_idx", []),
            "h_anchor_rule": meta.get("h_anchor_rule", "unknown"),
            "h_short_contact_count": meta.get("h_short_contact_count", np.nan),
            "h_extended_contact_count": meta.get("h_extended_contact_count", np.nan),
            "h_anchor_idx": meta.get("h_anchor_idx", []),
            "split": split,
        })

unknown_splits = sum(e["split"] == "unknown" for e in errors)
if unknown_splits:
    raise RuntimeError(
        f"Could not assign {unknown_splits} frames to the checkpoint split."
    )


# ============================================================
# ANCHOR-RULE DIAGNOSTICS
# ============================================================

print("\n=== O-ANCHOR EXTRACTION RULES ===")
o_rule_counts = defaultdict(int)
o_rule_abs_errors = defaultdict(list)

for e in errors:
    rule = e["o_anchor_rule"]
    o_rule_counts[rule] += 1
    o_rule_abs_errors[rule].append(e["abs_error"])

for rule in sorted(o_rule_counts):
    print(
        f"{rule}: n={o_rule_counts[rule]} | "
        f"MAE={np.mean(o_rule_abs_errors[rule]):.4f} eV"
    )

print("\n=== H-ANCHOR EXTRACTION RULES ===")
rule_counts = defaultdict(int)
rule_abs_errors = defaultdict(list)

for e in errors:
    rule = e["h_anchor_rule"]
    rule_counts[rule] += 1
    rule_abs_errors[rule].append(e["abs_error"])

for rule in sorted(rule_counts):
    print(
        f"{rule}: n={rule_counts[rule]} | "
        f"MAE={np.mean(rule_abs_errors[rule]):.4f} eV"
    )

# ============================================================
# METRICS
# ============================================================

y_true_energy = np.array(y_true_energy)
y_pred_energy = np.array(y_pred_energy)

mae = np.mean(np.abs(y_true_energy - y_pred_energy))
rmse = np.sqrt(np.mean((y_true_energy - y_pred_energy) ** 2))
signed_mean_error = np.mean(y_pred_energy - y_true_energy)

print("\n=== GLOBAL METRICS ===")
print(f"MAE:  {mae:.4f} eV")
print(f"RMSE: {rmse:.4f} eV")
print(f"Mean signed error: {signed_mean_error:+.4f} eV")
print(f"R^2:  {r2_score(y_true_energy, y_pred_energy):.4f}")


# ============================================================
# MAE PER GROUP
# ============================================================

group_errors = defaultdict(list)
group_signed = defaultdict(list)

for e in errors:
    group_errors[e["group"]].append(e["abs_error"])
    group_signed[e["group"]].append(e["error"])

print("\n=== MAE PER GROUP ===")
for group in sorted(group_errors.keys()):
    vals = group_errors[group]
    signed_vals = group_signed[group]
    print(
        f"{group}: "
        f"MAE = {np.mean(vals):.4f} eV | "
        f"Mean signed error = {np.mean(signed_vals):+.4f} eV | "
        f"n = {len(vals)}"
    )


# ============================================================
# MAE PER TRUE MOTIF
# ============================================================

motif_errors = defaultdict(list)
motif_signed = defaultdict(list)

for i in range(len(y_true_energy)):
    motif = id_to_motif[y_true_motif[i]]
    err = y_pred_energy[i] - y_true_energy[i]
    motif_errors[motif].append(abs(err))
    motif_signed[motif].append(err)

print("\n=== MAE PER TRUE MOTIF ===")
for motif in sorted(motif_errors.keys()):
    errs = motif_errors[motif]
    signed_errs = motif_signed[motif]
    print(
        f"{motif}: "
        f"MAE = {np.mean(errs):.4f} eV | "
        f"Mean signed error = {np.mean(signed_errs):+.4f} eV | "
        f"n = {len(errs)}"
    )

# ============================================================
# TRAIN/VALIDATION METRICS AND PARITY PLOT
# ============================================================

train_true = np.array([e["true_energy"] for e in errors if e["split"] == "train"])
train_pred = np.array([e["pred_energy"] for e in errors if e["split"] == "train"])

val_true = np.array([e["true_energy"] for e in errors if e["split"] == "validation"])
val_pred = np.array([e["pred_energy"] for e in errors if e["split"] == "validation"])


r2_train = r2_score(train_true, train_pred)
r2_val = r2_score(val_true, val_pred)

mae_train = np.mean(np.abs(train_true - train_pred))
rmse_train = np.sqrt(np.mean((train_true - train_pred) ** 2))

mae_val = np.mean(np.abs(val_true - val_pred))
rmse_val = np.sqrt(np.mean((val_true - val_pred) ** 2))

print(f"R^2 (train):   {r2_train:.4f}")
print(f"R^2 (val):     {r2_val:.4f}")
print(f"MAE (train):   {mae_train:.4f} eV")
print(f"MAE (val):     {mae_val:.4f} eV")
print(f"RMSE (train):  {rmse_train:.4f} eV")
print(f"RMSE (val):    {rmse_val:.4f} eV")

fig, ax = plt.subplots(figsize=(7, 7))

train_face = mcolors.to_rgba("forestgreen", alpha=0.3)
val_face = mcolors.to_rgba("darkorange", alpha=0.3)

ax.scatter(
    train_true, train_pred,
    s=70, linewidths=1.5,
    facecolor=train_face, edgecolor="darkgreen",
    label="Train",
)
ax.scatter(
    val_true, val_pred,
    s=70, linewidths=1.5,
    facecolor=val_face, edgecolor="chocolate",
    label="Validation",
)

min_e = min(y_true_energy.min(), y_pred_energy.min())
max_e = max(y_true_energy.max(), y_pred_energy.max())
padding = 0.05 * max(max_e - min_e, 1e-8)
lim_lo = min_e - padding
lim_hi = max_e + padding

ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="black", linewidth=1.5)
ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo, lim_hi)
ax.set_aspect("equal", adjustable="box")
ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

ax.set_xlabel(r"DFT $\mathrm{E}_{\mathrm{ads}}$ (eV)", fontsize=20)
ax.set_ylabel(r"Predicted $\mathrm{E}_{\mathrm{ads}}$ (eV)", fontsize=20)
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(which="major", length=8)
ax.tick_params(which="minor", length=4)
ax.tick_params(axis="both", labelsize=18)
ax.legend(fontsize=18)
fig.tight_layout()
fig.savefig(PARITY_FILE)
plt.close(fig)

# ============================================================
# O-H* DISTANCE ANALYSIS
# ============================================================

print(
    f"\n=== CONFIGURATIONS WITH O-H* DISTANCE < "
    f"{O_H_ADS_DISTANCE_THRESHOLD:.2f} A ===\n"
)

close_o_h_frames = []
for e in errors:
    distance = e["o_h_ads_distance"]
    if distance is not None and distance < O_H_ADS_DISTANCE_THRESHOLD:
        close_o_h_frames.append(e)
        print(
            f"global {e['global_idx']} | local {e['frame_idx']} | "
            f"O-H* distance = {distance:.3f} A | "
            f"error = {e['error']:+.3f} eV | "
            f"abs_error = {e['abs_error']:.3f} eV | "
            f"motif = {e['true_motif']} | "
            f"group = {e['group']}"
        )

print(
    f"\nTotal configurations with O-H* < {O_H_ADS_DISTANCE_THRESHOLD:.2f} A: "
    f"{len(close_o_h_frames)}"
)


# ============================================================
# SAVE WORST FRAMES TO CSV
# ============================================================

errors_sorted = sorted(errors, key=lambda x: x["abs_error"], reverse=True)

# ============================================================
# SAVE WORST STRUCTURES
# ============================================================

os.makedirs(WORST_STRUCTURES_DIR, exist_ok=True)

for rank, e in enumerate(errors_sorted[:N_WORST_STRUCTURES], start=1):
    idx = e["global_idx"]
    atoms = frames[idx].copy()

    atoms.info["error_rank"] = rank
    atoms.info["global_idx"] = idx
    atoms.info["local_frame_idx"] = e["frame_idx"]
    atoms.info["true_energy_eV"] = float(e["true_energy"])
    atoms.info["pred_energy_eV"] = float(e["pred_energy"])
    atoms.info["signed_error_eV"] = float(e["error"])
    atoms.info["abs_error_eV"] = float(e["abs_error"])
    atoms.info["true_motif"] = e["true_motif"]
    atoms.info["group"] = e["group"]
    atoms.info["o_h_ads_distance_A"] = (
        float(e["o_h_ads_distance"])
        if e["o_h_ads_distance"] is not None
        else np.nan
    )
    atoms.info["o_anchor_rule"] = e["o_anchor_rule"]
    atoms.info["o_short_contact_count"] = int(e["o_short_contact_count"])
    atoms.info["o_extended_contact_count"] = int(e["o_extended_contact_count"])
    atoms.info["o_anchor_idx"] = list(map(int, e["o_anchor_idx"]))
    atoms.info["h_anchor_rule"] = e["h_anchor_rule"]
    atoms.info["h_short_contact_count"] = int(e["h_short_contact_count"])
    atoms.info["h_extended_contact_count"] = int(e["h_extended_contact_count"])
    atoms.info["h_anchor_idx"] = list(map(int, e["h_anchor_idx"]))
    atoms.info["split"] = e["split"]

    filename = (
        f"rank_{rank:02d}"
        f"_global_{idx}"
        f"_local_{e['frame_idx']}"
        f"_motif_{e['true_motif']}"
        f"_group_{e['group']}.extxyz"
    )

    write(os.path.join(WORST_STRUCTURES_DIR, filename), atoms, format="extxyz")

print(
    f"\nSaved the {N_WORST_STRUCTURES} largest-error structures in: "
    f"{WORST_STRUCTURES_DIR}"
)

with open(WORST_TABLE_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "rank",
        "global_idx",
        "local_frame_idx",
        "split",
        "signed_error_eV",
        "abs_error_eV",
        "true_energy_eV",
        "pred_energy_eV",
        "true_motif",
        "group",
        "o_h_ads_distance_A",
        "o_anchor_rule",
        "o_short_contact_count",
        "o_extended_contact_count",
        "o_anchor_idx",
        "h_anchor_rule",
        "h_short_contact_count",
        "h_extended_contact_count",
        "h_anchor_idx",
    ])

    for rank, e in enumerate(errors_sorted[:N_WORST_TABLE], start=1):
        writer.writerow([
            rank,
            e["global_idx"],
            e["frame_idx"],
            e["split"],
            f"{e['error']:.6f}",
            f"{e['abs_error']:.6f}",
            f"{e['true_energy']:.6f}",
            f"{e['pred_energy']:.6f}",
            e["true_motif"],
            e["group"],
            (
                f"{e['o_h_ads_distance']:.6f}"
                if e["o_h_ads_distance"] is not None
                else "NA"
            ),
            e["o_anchor_rule"],
            e["o_short_contact_count"],
            e["o_extended_contact_count"],
            str(e["o_anchor_idx"]),
            e["h_anchor_rule"],
            e["h_short_contact_count"],
            e["h_extended_contact_count"],
            str(e["h_anchor_idx"]),
        ])

print(f"\nSaved the {N_WORST_TABLE} largest errors to: {WORST_TABLE_FILE}")


# ============================================================
# SAVE FULL ERROR TABLE TO CSV
# ============================================================

with open(ERROR_TABLE_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "global_idx",
        "frame_idx",
        "split",
        "true_energy_eV",
        "pred_energy_eV",
        "signed_error_eV",
        "abs_error_eV",
        "group",
        "true_motif",
        "o_h_ads_distance_A",
        "o_anchor_rule",
        "o_short_contact_count",
        "o_extended_contact_count",
        "o_anchor_idx",
        "h_anchor_rule",
        "h_short_contact_count",
        "h_extended_contact_count",
        "h_anchor_idx",
    ])

    for e in errors_sorted:
        writer.writerow([
            e["global_idx"],
            e["frame_idx"],
            e["split"],
            f"{e['true_energy']:.6f}",
            f"{e['pred_energy']:.6f}",
            f"{e['error']:.6f}",
            f"{e['abs_error']:.6f}",
            e["group"],
            e["true_motif"],
            (
                f"{e['o_h_ads_distance']:.6f}"
                if e["o_h_ads_distance"] is not None
                else "NA"
            ),
            e["o_anchor_rule"],
            e["o_short_contact_count"],
            e["o_extended_contact_count"],
            str(e["o_anchor_idx"]),
            e["h_anchor_rule"],
            e["h_short_contact_count"],
            e["h_extended_contact_count"],
            str(e["h_anchor_idx"]),
        ])

print(f"Saved full error table to: {ERROR_TABLE_FILE}")
print(f"Saved parity plot to: {PARITY_FILE}")
