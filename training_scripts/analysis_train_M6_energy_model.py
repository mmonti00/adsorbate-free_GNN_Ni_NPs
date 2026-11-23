#!/usr/bin/env python3
"""Evaluate a trained M6 model and create parity plots.

The script imports ``train_M6_energy_model.py`` so that evaluation reconstructs
exactly the same nanoparticle and periodic-slab graphs used for training. It
checks the preprocessing settings stored in the checkpoint, restores the saved
domain-specific train/validation splits, reports MAE, RMSE, and R2, and writes:

* nanoparticle-versus-surface and train-versus-validation parity plots;
* validation parity grids for each bare Ni structure, using absolute and
  relative adsorption energies;
* a CSV table containing every target and prediction; and
* per-configuration metric tables for the validation grids.

For relative energies, both DFT and predicted energies are referenced to the
same configuration: the DFT minimum within each split/domain/bare-structure
group. Run this script from its own directory so the imported training module
and relative paths resolve as provided.

Outputs are written to ``check_train_M6_energy_model/``.
"""

import importlib
import os
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from sklearn.metrics import r2_score

# ============================================================
# INPUTS AND OUTPUTS
# ============================================================

# Training module name without the .py suffix.
TRAIN_MODULE = "train_M6_energy_model"

OUTDIR = "check_train_M6_energy_model"
os.makedirs(OUTDIR, exist_ok=True)

DOMAIN_PARITY_FILE = "parity_domain_NPs_Surfaces.pdf"
SPLIT_PARITY_FILE = "parity_train_validation.pdf"
ABSOLUTE_GRID_FILE = "parity_validation_by_config_absolute.pdf"
RELATIVE_GRID_FILE = "parity_validation_by_config_relative.pdf"
ERROR_TABLE_FILE = "model_errors.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

# ============================================================
# TRAINING MODULE
# ============================================================

tr = importlib.import_module(TRAIN_MODULE)

print(f"Training module: {TRAIN_MODULE} ({getattr(tr, '__file__', '?')})")

BareSiteGCNN = tr.BareSiteGCNN
DEVICE = tr.DEVICE
# Checkpoint to evaluate. The default is the released M6 model, which reproduces
# the validation metrics reported in the paper. To evaluate a checkpoint written
# by train_M6_energy_model.py in this directory, set MODEL_PATH = tr.MODEL_OUT.
MODEL_PATH = "../models_checkpoints/M6_energy_model.pt"
initial_refs = tr.load_initial_reference_structures()

# ============================================================
# DATASET REBUILDING
# ============================================================


def load_domain_dataset(domain_name, extxyz_file, sample_weight):
    """Rebuild one domain using the graph pipeline from the training module."""
    if extxyz_file is None:
        return [], [], [], [], [], []

    return tr.build_pyg_dataset(
        extxyz_file,
        initial_refs,
        domain_label=domain_name,
        sample_weight=sample_weight,
    )


def combine_domains():
    """Collect NP and slab graphs with domain-local and global indices."""
    all_rows = []
    domain_specs = [("NP", tr.NP_EXTXYZ_FILE, tr.NP_SAMPLE_WEIGHT)]

    if getattr(tr, "SLAB_EXTXYZ_FILE", None) is not None:
        domain_specs.append(("SLAB", tr.SLAB_EXTXYZ_FILE, tr.SLAB_SAMPLE_WEIGHT))

    for domain_name, extxyz_file, sample_weight in domain_specs:
        (
            dataset,
            _metadata,
            _frames,
            group_labels,
            _motif_labels,
            _domain_labels,
        ) = load_domain_dataset(domain_name, extxyz_file, sample_weight)

        for local_i, (data, group) in enumerate(zip(dataset, group_labels)):
            all_rows.append(
                {
                    "domain": domain_name,
                    "domain_local_idx": local_i,
                    "dataset_global_idx": len(all_rows),
                    "data": data,
                    "group": group,
                }
            )

    return all_rows


rows = combine_domains()
if not rows:
    raise RuntimeError("No frames were loaded. Check NP_EXTXYZ_FILE / SLAB_EXTXYZ_FILE.")

# ============================================================
# CHECKPOINT AND SAVED SPLITS
# ============================================================

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

settings = ckpt.get("settings", {})
expected_settings = {
    "OH_BOND_CUTOFF": tr.OH_BOND_CUTOFF,
    "NI_O_CUTOFF": tr.NI_O_CUTOFF,
    "NI_O_FOLDED_CUTOFF": tr.NI_O_FOLDED_CUTOFF,
    "O_FOLDED_N_SHORT": tr.O_FOLDED_N_SHORT,
    "O_FOLDED_N_TOTAL": tr.O_FOLDED_N_TOTAL,
    "MAX_STANDARD_O_ANCHORS": tr.MAX_STANDARD_O_ANCHORS,
    "NI_H_CUTOFF": tr.NI_H_CUTOFF,
    "NI_H_FOLDED_CUTOFF": tr.NI_H_FOLDED_CUTOFF,
    "H_FOLDED_N_SHORT": tr.H_FOLDED_N_SHORT,
    "H_FOLDED_N_TOTAL": tr.H_FOLDED_N_TOTAL,
    "MAX_STANDARD_H_ANCHORS": tr.MAX_STANDARD_H_ANCHORS,
    "LOCAL_CUTOFF": tr.LOCAL_CUTOFF,
    "EDGE_CUTOFF": tr.EDGE_CUTOFF,
    "N_RBF": tr.N_RBF,
    "RBF_DMIN": tr.RBF_DMIN,
    "RBF_DMAX": tr.RBF_DMAX,
    "RBF_GAMMA": tr.RBF_GAMMA,
}

print("\n=== CHECKPOINT PREPROCESSING CHECK ===")
for key, expected in expected_settings.items():
    if key not in settings:
        raise RuntimeError(
            f"Checkpoint is missing {key!r}. "
            "It may not have been produced by the hybrid O/H fine-tuning script."
        )
    actual = settings[key]
    if float(actual) != float(expected):
        raise RuntimeError(
            f"Checkpoint/module mismatch for {key}: "
            f"checkpoint={actual}, imported module={expected}."
        )
    print(f"{key} = {actual}")

for key, expected in {
    "LOCAL_CUTOFF_BY_DOMAIN": tr.LOCAL_CUTOFF_BY_DOMAIN,
    "EDGE_CUTOFF_BY_DOMAIN": tr.EDGE_CUTOFF_BY_DOMAIN,
}.items():
    actual = settings.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint/module mismatch for {key}: "
            f"checkpoint={actual}, imported module={expected}."
        )
    print(f"{key} = {actual}")

if list(settings.get("SITE_TYPES", [])) != list(tr.SITE_TYPES):
    raise RuntimeError(
        "Checkpoint/module mismatch for SITE_TYPES: "
        f"checkpoint={settings.get('SITE_TYPES')}, imported module={list(tr.SITE_TYPES)}."
    )

if settings.get("TARGET_REFERENCE") != "H2O_gas":
    raise RuntimeError(
        "Checkpoint does not use the expected H2O-referenced adsorption energies."
    )

if not bool(ckpt.get("energy_only", False)):
    raise RuntimeError(
        "The checkpoint is not marked as energy-only. "
        "Use the checkpoint produced by the energy-only fine-tuning script."
    )

model = BareSiteGCNN(
    node_dim=ckpt["node_dim"],
    edge_dim=ckpt["edge_dim"],
    hidden_dim=ckpt["hidden_dim"],
).to(DEVICE)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

ID_TO_MOTIF = ckpt.get("id_to_motif", tr.ID_TO_MOTIF)
domain_splits = ckpt.get("domain_splits")
if not domain_splits:
    raise RuntimeError(
        "Checkpoint does not contain domain-specific train/validation indices."
    )

split_indices = {}
for domain, split in domain_splits.items():
    train_indices = set(split.get("train_idx", []))
    validation_indices = set(split.get("val_idx", []))
    overlap = train_indices & validation_indices
    if overlap:
        raise RuntimeError(
            f"Checkpoint split for {domain} contains {len(overlap)} overlapping indices."
        )
    split_indices[domain] = {
        "train": train_indices,
        "validation": validation_indices,
    }


def infer_split(row):
    """Return the checkpoint split for one domain-local frame index."""
    domain = row["domain"]
    local_idx = int(row["domain_local_idx"])
    if domain not in split_indices:
        return "unknown"

    if local_idx in split_indices[domain]["train"]:
        return "train"
    if local_idx in split_indices[domain]["validation"]:
        return "validation"
    return "unknown"

# ============================================================
# INFERENCE
# ============================================================

records = []

with torch.no_grad():
    for row in rows:
        data = row["data"].to(DEVICE)
        batch_vec = torch.zeros(data.x.shape[0], dtype=torch.long, device=DEVICE)

        e_pred = model(data.x, data.edge_index, data.edge_attr, batch_vec)

        e_pred = float(e_pred.item())
        e_true = float(data.y_energy.item())

        records.append({
            "dataset_global_idx": int(row["dataset_global_idx"]),
            "domain": row["domain"],
            "domain_local_idx": int(row["domain_local_idx"]),
            "group": row["group"],
            "motif": ID_TO_MOTIF[int(data.y_motif.item())],
            "split": infer_split(row),
            "true_energy": e_true,
            "pred_energy": e_pred,
        })

df = pd.DataFrame(records)
unknown_splits = int((df["split"] == "unknown").sum())
if unknown_splits:
    raise RuntimeError(
        f"Could not recover the saved split for {unknown_splits} configurations."
    )
df.to_csv(os.path.join(OUTDIR, ERROR_TABLE_FILE), index=False)

# ============================================================
# METRICS
# ============================================================


def safe_r2(y_true, y_pred):
    """Return R2, or NaN when fewer than two distinct targets are present."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return np.nan
    return float(r2_score(y_true, y_pred))


def print_metrics(title, sub, true_col="true_energy", pred_col="pred_energy"):
    """Print sample count, MAE, RMSE, and R2 for one dataframe subset."""
    print(f"\n=== {title} ===")
    true = sub[true_col].to_numpy(float)
    pred = sub[pred_col].to_numpy(float)
    if len(true) == 0:
        print("  n=0")
        return
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = safe_r2(true, pred)
    print(f"n:    {len(true)}")
    print(f"MAE:  {mae:.4f} eV")
    print(f"RMSE: {rmse:.4f} eV")
    print(f"R^2:  {r2:.4f}" if not np.isnan(r2) else "R^2:  NA")


for label, sub in [
    ("ALL", df),
    ("TRAIN", df[df["split"] == "train"]),
    ("VALIDATION", df[df["split"] == "validation"]),
    ("NP", df[df["domain"] == "NP"]),
    ("SLAB (Surfaces)", df[df["domain"] == "SLAB"]),
    ("NP - TRAIN", df[(df["domain"] == "NP") & (df["split"] == "train")]),
    ("NP - VALIDATION", df[(df["domain"] == "NP") & (df["split"] == "validation")]),
    ("SLAB (Surfaces) - TRAIN", df[(df["domain"] == "SLAB") & (df["split"] == "train")]),
    ("SLAB (Surfaces) - VALIDATION", df[(df["domain"] == "SLAB") & (df["split"] == "validation")]),
]:
    print_metrics(label, sub)

# ============================================================
# PARITY-PLOT HELPERS
# ============================================================


def compute_limits(*arrays):
    """Return common square parity limits with five percent padding."""
    finite = [np.asarray(a, dtype=float) for a in arrays if len(a)]
    lo = min(float(np.min(a)) for a in finite)
    hi = max(float(np.max(a)) for a in finite)
    pad = 0.05 * max(hi - lo, 1e-8)
    return lo - pad, hi + pad


def style_parity_axis(ax, lim_lo, lim_hi, xlabel, ylabel):
    """Apply the shared parity-plot axes, ticks, and identity line."""
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="black", linewidth=1.5)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    # Identical locators and limits keep the x and y ticks matched.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="minor", length=4)
    ax.tick_params(axis="both", labelsize=20)


# ============================================================
# PER-CONFIGURATION COLORS AND LABELS
# ============================================================

NP_FACE_COLORS = {
    "Ni25":       "#3F7BD9",  # blue
    "Ni30":       "#7E57C2",  # violet
    "Ni35":       "#D95FBC",  # magenta
    "Ni40":       "#E64B5D",  # coral
    "Ni45":       "#F06FA9",  # pink
    "Ni50":       "#008C95",  # teal
    "Ni47_conf1": "#5B8DB8",  # blue-grey
    "Ni47_conf2": "#2F688E",  # darker blue-grey
}

SURFACE_FACE_COLORS = {
    "Ni111_2x2": "#3FA34D",  # green
    "Ni111_3x3": "#6FBE7A",  # lighter green
    "Ni111_4x3": "#A2D8A5",  # pale green

    "Ni100_3x3": "#E67E22",  # orange
    "Ni100_4x3": "#F0A35A",  # lighter orange
    "Ni100_4x4": "#F6C28B",  # pale orange
}

DEFAULT_GROUP_FACE_COLOR = "#A8CAC7"
GROUP_EDGE_DARKEN_FACTOR = 0.58

# Map bare-reference keys from the training module to manuscript labels.
# Single-structure NPs omit ``_conf1``; distinct Ni47 isomers retain it.
NP_GROUP_TO_LABEL = {
    "Ni25_conf1": "Ni25",
    "Ni30_conf1": "Ni30",
    "Ni35_conf1": "Ni35",
    "Ni40_conf1": "Ni40",
    "Ni45_conf1": "Ni45",
    "Ni47_conf1": "Ni47_conf1",
    "Ni47_conf2": "Ni47_conf2",
    "Ni50_conf1": "Ni50",
    "Ni55_conf1": "Ni55",
}

SLAB_GROUP_TO_LABEL = {
    "Ni111_conf1": "Ni111_2x2",
    "Ni111_conf2": "Ni111_3x3",
    "Ni111_conf3": "Ni111_4x3",
    "Ni100_conf1": "Ni100_3x3",
    "Ni100_conf2": "Ni100_4x3",
    "Ni100_conf3": "Ni100_4x4",
}


def darken_color(color, factor=GROUP_EDGE_DARKEN_FACTOR):
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return mcolors.to_hex(np.clip(rgb * factor, 0.0, 1.0))


def group_display_label(domain, group):
    if domain == "NP":
        return NP_GROUP_TO_LABEL.get(group, group)
    return SLAB_GROUP_TO_LABEL.get(group, group)


def group_plot_colors(domain, group):
    """Return (face_color, edge_color, display_label) for one Ni configuration."""
    label = group_display_label(domain, group)
    palette = NP_FACE_COLORS if domain == "NP" else SURFACE_FACE_COLORS
    face = palette.get(label, DEFAULT_GROUP_FACE_COLOR)
    return face, darken_color(face), label


def format_config_label(label):
    """Convert an internal group label into a manuscript-style panel title."""
    m = re.fullmatch(r"Ni(\d{3})_(\d+)x(\d+)", label)
    if m:
        facet, nx, ny = m.groups()
        return rf"Ni ({facet}) - {nx}x{ny}"

    m = re.fullmatch(r"Ni(\d+)_conf(\d+)", label)
    if m:
        n, iso = m.groups()
        return rf"Ni$_{{{n}}}$-iso{iso}"

    m = re.fullmatch(r"Ni(\d+)", label)
    if m:
        return rf"Ni$_{{{m.group(1)}}}$"

    return label


# Place nanoparticles first, followed by periodic surfaces.
_GROUP_ORDER_LABELS = list(NP_FACE_COLORS) + list(SURFACE_FACE_COLORS)


def ordered_groups(plot_df):
    """Return present (domain, group) pairs in the requested panel order."""
    pairs = list(
        plot_df[["domain", "group"]].drop_duplicates().itertuples(index=False, name=None)
    )
    labels = {(d, g): group_display_label(d, g) for d, g in pairs}

    def sort_key(item):
        label = labels[item]
        try:
            rank = _GROUP_ORDER_LABELS.index(label)
        except ValueError:
            rank = len(_GROUP_ORDER_LABELS)
        return (rank, label)

    return sorted(pairs, key=sort_key)


def plot_validation_group_grid(plot_df, true_col, pred_col, xlabel, ylabel, outfile, pairs=None):
    """Plot one validation parity panel per bare Ni structure.

    Each panel has its own square x/y range. Passing ``pairs`` fixes the panel
    layout so absolute and relative figures use identical geometry. The
    function returns the panel order for reuse by a second call.
    """
    if pairs is None:
        pairs = ordered_groups(plot_df)
    if not pairs:
        return pairs

    ncols = min(4, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()

    metrics_rows = []
    for panel_idx, (ax, (domain, group)) in enumerate(zip(axes.flat, pairs)):
        sub = plot_df[(plot_df["domain"] == domain) & (plot_df["group"] == group)]
        x = sub[true_col].to_numpy(float)
        y = sub[pred_col].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) == 0:
            continue
        ax.set_axis_on()

        face, edge, label = group_plot_colors(domain, group)
        face_rgba = mcolors.to_rgba(face, alpha=0.5)

        ax.scatter(
            x, y,
            s=65, linewidths=1.4,
            facecolor=face_rgba, edgecolor=edge,
            zorder=3,
        )

        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        pad = 0.08 * max(hi - lo, 1e-8)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        # Match x and y ticks within each square panel.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

        ax.set_title(format_config_label(label), fontsize=24)
        ax.set_xlabel(xlabel, fontsize=22)
        # The y label is shared along each row, so draw it on the first column only.
        if panel_idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=22)
        # Panel letter at the top-left corner of the panel's grid cell, as in
        # the manuscript figures (matplotlib cannot select the bold face of the
        # macOS Helvetica collection, so Arial Bold is used).
        fig_w, fig_h = fig.get_size_inches()
        row, col = divmod(panel_idx, ncols)
        fig.text(
            col / ncols + 0.02 / fig_w, 1.0 - row / nrows - 0.22 / fig_h,
            f"{chr(ord('a') + panel_idx)})",
            fontsize=25, fontweight="bold",
            fontfamily=["Arial", "DejaVu Sans"], ha="left", va="top",
        )
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", length=8, labelsize=18)
        ax.tick_params(which="minor", length=4)
        ax.grid(True, alpha=0.3)

        err = y - x
        r2 = safe_r2(x, y)
        metrics_rows.append({
            "domain": domain, "group": group, "label": label,
            "n": len(x),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": r2,
        })

    # Align labels despite differing tick-label widths.
    fig.align_xlabels(list(axes.flat))
    fig.align_ylabels(list(axes.flat))

    # Fixed margins give the absolute and relative grids identical geometry.
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.07, wspace=0.38, hspace=0.42)
    fig.savefig(os.path.join(OUTDIR, outfile))
    plt.close(fig)

    stem, _ = os.path.splitext(outfile)
    metrics_path = os.path.join(OUTDIR, f"{stem}_metrics.txt")
    with open(metrics_path, "w") as f:
        header = f"{'domain':<8}{'group':<16}{'label':<14}{'n':>6}{'MAE':>10}{'RMSE':>10}{'R2':>10}\n"
        f.write(header)
        f.write("-" * (len(header) - 1) + "\n")
        for r in metrics_rows:
            r2_str = f"{r['R2']:.4f}" if not np.isnan(r["R2"]) else "NA"
            f.write(
                f"{r['domain']:<8}{r['group']:<16}{r['label']:<14}{r['n']:>6}"
                f"{r['MAE']:>10.4f}{r['RMSE']:>10.4f}{r2_str:>10}\n"
            )

    return pairs


# ============================================================
# PARITY PLOT 1: NPs vs Surfaces
# ============================================================

np_true = df.loc[df["domain"] == "NP", "true_energy"].to_numpy(float)
np_pred = df.loc[df["domain"] == "NP", "pred_energy"].to_numpy(float)
slab_true = df.loc[df["domain"] == "SLAB", "true_energy"].to_numpy(float)
slab_pred = df.loc[df["domain"] == "SLAB", "pred_energy"].to_numpy(float)

fig, ax = plt.subplots(figsize=(7, 7))

np_face = mcolors.to_rgba("royalblue", alpha=0.3)
slab_face = mcolors.to_rgba("firebrick", alpha=0.3)

ax.scatter(
    np_true, np_pred,
    s=70, linewidths=1.5,
    facecolor=np_face, edgecolor="navy",
    label="Nanoparticles",
)
ax.scatter(
    slab_true, slab_pred,
    s=70, linewidths=1.5,
    facecolor=slab_face, edgecolor="darkred",
    label="Surfaces",
)

lim_lo, lim_hi = compute_limits(np_true, np_pred, slab_true, slab_pred)
style_parity_axis(
    ax,
    lim_lo,
    lim_hi,
    r"DFT $\mathrm{E}_{\mathrm{ads}}$ (eV)",
    r"Predicted $\mathrm{E}_{\mathrm{ads}}$ (eV)",
)
ax.legend(fontsize=19)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, DOMAIN_PARITY_FILE))
plt.close(fig)

# ============================================================
# PARITY PLOT 2: Train vs Validation
# ============================================================

train_true = df.loc[df["split"] == "train", "true_energy"].to_numpy(float)
train_pred = df.loc[df["split"] == "train", "pred_energy"].to_numpy(float)
val_true = df.loc[df["split"] == "validation", "true_energy"].to_numpy(float)
val_pred = df.loc[df["split"] == "validation", "pred_energy"].to_numpy(float)

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

lim_lo, lim_hi = compute_limits(train_true, train_pred, val_true, val_pred)
style_parity_axis(
    ax,
    lim_lo,
    lim_hi,
    r"DFT $\mathrm{E}_{\mathrm{ads}}$ (eV)",
    r"Predicted $\mathrm{E}_{\mathrm{ads}}$ (eV)",
)
ax.legend(fontsize=19)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, SPLIT_PARITY_FILE))
plt.close(fig)

# ============================================================
# PARITY PLOTS 3 & 4: validation, one panel per Ni configuration
# ============================================================

df["dft_rel_eV"] = np.nan
df["pred_rel_to_dft_min_eV"] = np.nan

# Reference both energy series to the configuration with the lowest DFT
# energy in each split/domain/bare-structure group.
for (split, domain, group), sub in df.groupby(["split", "domain", "group"]):
    if len(sub) < 2:
        continue
    idx_min = sub["true_energy"].idxmin()
    e_true_ref = float(df.loc[idx_min, "true_energy"])
    e_pred_ref = float(df.loc[idx_min, "pred_energy"])
    df.loc[sub.index, "dft_rel_eV"] = sub["true_energy"] - e_true_ref
    df.loc[sub.index, "pred_rel_to_dft_min_eV"] = sub["pred_energy"] - e_pred_ref

plot_df = df[df["split"] == "validation"].copy()
if plot_df.empty:
    plot_df = df.copy()

print_metrics("VALIDATION - ABSOLUTE E_ads", plot_df, "true_energy", "pred_energy")

# Reuse the absolute-grid panel order for identical figure geometry.
plot_df_rel = plot_df.dropna(subset=["dft_rel_eV", "pred_rel_to_dft_min_eV"]).copy()
print_metrics(
    "VALIDATION - RELATIVE E_ads (per-group, vs. DFT min)",
    plot_df_rel, "dft_rel_eV", "pred_rel_to_dft_min_eV",
)

grid_pairs = plot_validation_group_grid(
    plot_df, "true_energy", "pred_energy",
    r"DFT $\mathrm{E}_{\mathrm{ads}}$ (eV)",
    r"Predicted $\mathrm{E}_{\mathrm{ads}}$ (eV)",
    ABSOLUTE_GRID_FILE,
)

plot_validation_group_grid(
    plot_df_rel, "dft_rel_eV", "pred_rel_to_dft_min_eV",
    r"DFT $\Delta\mathrm{E}_{\mathrm{ads}}$ (eV)",
    r"Predicted $\Delta\mathrm{E}_{\mathrm{ads}}$ (eV)",
    RELATIVE_GRID_FILE,
    pairs=grid_pairs,
)

print(f"\nSaved plots in: {OUTDIR}")
print(f"  {OUTDIR}/{DOMAIN_PARITY_FILE}")
print(f"  {OUTDIR}/{SPLIT_PARITY_FILE}")
print(f"  {OUTDIR}/{ABSOLUTE_GRID_FILE} (+ _metrics.txt)")
print(f"  {OUTDIR}/{RELATIVE_GRID_FILE} (+ _metrics.txt)")
print(f"  {OUTDIR}/{ERROR_TABLE_FILE}")
