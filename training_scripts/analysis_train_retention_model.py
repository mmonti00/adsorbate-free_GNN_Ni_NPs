#!/usr/bin/env python3
"""Analyze the trained binary NP/slab motif-retention classifier.

The analysis rebuilds the datasets with the same functions used for training,
including the OH*+H* validity filter and the header-defined initial candidates.
It then restores the saved train/validation split and reports performance for
all structures and for each domain separately.

Binary target:
    1 = retained_exact
    0 = migrated

Model architecture:
    shared message-passing encoder
        -> O-focused and H-focused weighted pooling
        -> NP or slab binary retention head

Run this script from its own directory so the training module, datasets, and
checkpoint resolve through the relative paths below.
"""

import csv
import importlib
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

from ase.io import read

# Same typography convention as the other paper-quality plots in this project.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    }
)

# Precision-recall curve colors, one per selector.
PR_CURVE_COLORS = {
    "ALL": "teal",
    "NP": "navy",
    "SLAB": "darkred",
}

# Train/validation colors used consistently across the diagnostic plots.
TRAIN_FACE = mcolors.to_rgba("forestgreen", alpha=0.3)
TRAIN_EDGE = "darkgreen"
VAL_FACE = mcolors.to_rgba("darkorange", alpha=0.3)
VAL_EDGE = "chocolate"


# ============================================================
# TRAINING MODULE
# ============================================================

TRAINING_MODULE = "train_retention_classifier"

train_mod = importlib.import_module(TRAINING_MODULE)

MotifRetentionTwoHeadGNN = train_mod.MotifRetentionTwoHeadGNN
build_dataset = train_mod.build_dataset
build_group_labels = train_mod.build_group_labels
DOMAIN_TO_ID = train_mod.DOMAIN_TO_ID
DEVICE = train_mod.DEVICE

NP_INITIAL_EXTXYZ_FILE = train_mod.NP_INITIAL_EXTXYZ_FILE
NP_FINAL_EXTXYZ_FILE = train_mod.NP_FINAL_EXTXYZ_FILE
SLAB_INITIAL_EXTXYZ_FILE = train_mod.SLAB_INITIAL_EXTXYZ_FILE
SLAB_FINAL_EXTXYZ_FILE = train_mod.SLAB_FINAL_EXTXYZ_FILE

MODEL_OUT = train_mod.MODEL_OUT
OH_BOND_CUTOFF = train_mod.OH_BOND_CUTOFF
MIN_OH_SEPARATION = train_mod.MIN_OH_SEPARATION
USE_HEADER_INITIAL_SITE = train_mod.USE_HEADER_INITIAL_SITE


# ============================================================
# SETTINGS
# ============================================================

# Checkpoint to evaluate. The default is the released retention model; to
# evaluate a checkpoint written by train_retention_classifier.py in this
# directory, set MODEL_PATH = MODEL_OUT.
MODEL_PATH = "../models_checkpoints/retention_model.pt"

DOMAIN_FILES = {
    "NP": (NP_INITIAL_EXTXYZ_FILE, NP_FINAL_EXTXYZ_FILE),
    "SLAB": (SLAB_INITIAL_EXTXYZ_FILE, SLAB_FINAL_EXTXYZ_FILE),
}

OUTPUT_TAG = "local_pool"
FIGURE_FORMAT = "pdf"
SCORE_PLOT_THRESHOLD = 0.80

OUT_CSV = f"motif_retention_binary_predictions_{OUTPUT_TAG}.csv"

THRESHOLD_SCAN_FILES = {
    "ALL": f"motif_retention_threshold_scan_validation_ALL_{OUTPUT_TAG}.csv",
    "NP": f"motif_retention_threshold_scan_validation_NP_{OUTPUT_TAG}.csv",
    "SLAB": f"motif_retention_threshold_scan_validation_SLAB_{OUTPUT_TAG}.csv",
}

THRESHOLDS = [0.50, 0.75, 0.80, 0.85, 0.90]
MOTIF_REPORT_THRESHOLD = 0.80


# ============================================================
# METRICS
# ============================================================


def safe_auc(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, score))


def safe_ap(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")

    return float(average_precision_score(y_true, score))


def binary_metrics(y_true, p_retained, threshold=0.50):
    y_true = np.asarray(y_true, dtype=int)
    p_retained = np.asarray(p_retained, dtype=float)

    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "n": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "AP": float("nan"),
            "ROC_AUC": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "n_pred_retained": 0,
        }

    y_pred = (p_retained >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    # Rows = true [retained, migrated]
    # Columns = predicted [retained, migrated]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "AP": safe_ap(y_true, p_retained),
        "ROC_AUC": safe_auc(y_true, p_retained),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pred_retained": int(y_pred.sum()),
    }


def print_binary_report(name, y_true, p_retained, threshold):
    m = binary_metrics(y_true, p_retained, threshold)

    print(
        f"  {name:<14s} threshold={threshold:.2f} "
        f"acc={m['accuracy']:.4f} "
        f"bal={m['balanced_accuracy']:.4f} "
        f"prec={m['precision']:.4f} "
        f"rec={m['recall']:.4f} "
        f"spec={m['specificity']:.4f} "
        f"f1={m['f1']:.4f} "
        f"AP={m['AP']:.4f} "
        f"ROC-AUC={m['ROC_AUC']:.4f} "
        f"n={m['n']}"
    )

    print("    confusion [retained=1, migrated=0]:")
    print("                       pred_retained  pred_migrated")
    print(f"    true_retained       {m['tp']:>6d}        {m['fn']:>6d}")
    print(f"    true_migrated   {m['fp']:>6d}        {m['tn']:>6d}")


def calibration_bins(y_true, p_retained, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    p_retained = np.asarray(p_retained, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (p_retained >= lo) & (p_retained <= hi)
        else:
            mask = (p_retained >= lo) & (p_retained < hi)

        if not mask.any():
            continue

        rows.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(mask.sum()),
                "mean_score": float(p_retained[mask].mean()),
                "emp_retained": float(y_true[mask].mean()),
            }
        )

    return rows


def save_threshold_scan(y_true, p_retained, path):
    with open(path, "w", newline="") as handle:
        fieldnames = [
            "threshold",
            "n",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "AP",
            "ROC_AUC",
            "tp",
            "fp",
            "tn",
            "fn",
            "n_pred_retained",
        ]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for threshold in np.linspace(0.0, 1.0, 101):
            writer.writerow(
                binary_metrics(
                    y_true,
                    p_retained,
                    threshold=float(threshold),
                )
            )


# ============================================================
# PLOTS
# ============================================================


def plot_retention_score(rows, selector_name):
    selected = [
        row for row in rows if selector_name == "ALL" or row["domain"] == selector_name
    ]

    if not selected:
        return

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7, 5))

    for split, face, edge in [
        ("train", TRAIN_FACE, TRAIN_EDGE),
        ("validation", VAL_FACE, VAL_EDGE),
    ]:
        subset = [row for row in selected if row["split"] == split]

        if not subset:
            continue

        x = np.array(
            [row["p_retained"] for row in subset],
            dtype=float,
        )
        y = np.array(
            [row["true_retained"] for row in subset],
            dtype=float,
        )

        y += rng.uniform(-0.035, 0.035, size=len(y))

        ax.scatter(
            x,
            y,
            s=28,
            linewidths=1.0,
            facecolor=face,
            edgecolor=edge,
            label=split,
        )

    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)
    ax.axvline(
        SCORE_PLOT_THRESHOLD,
        linestyle="--",
        color="black",
        label=f"threshold={SCORE_PLOT_THRESHOLD:.2f}",
    )

    # Fixed axis limits (not autoscaled) so the legend below sits at the
    # same place relative to the data on every plot.
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.15)

    ax.set_xlabel(r"$s_{\mathrm{ret}}$", fontsize=15)
    ax.set_ylabel("Reference class", fontsize=15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["migrated", "retained"])
    ax.set_title(f"Retained score: {selector_name} — O/H pooling")

    ax.tick_params(axis="both", which="major", length=7, labelsize=14)

    # Keep the legend above the negative-class cluster.
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.20), fontsize=13)

    plt.tight_layout()

    path = f"motif_retention_score_{selector_name}_{OUTPUT_TAG}.{FIGURE_FORMAT}"

    plt.savefig(path, dpi=600)
    plt.close()

    print(f"Saved {path}")


def plot_calibration(y_true, p_retained, selector_name):
    cal = calibration_bins(
        y_true,
        p_retained,
        n_bins=10,
    )

    if not cal:
        print(f"Calibration plot skipped for {selector_name}: no bins.")
        return

    mean_scores = np.array([row["mean_score"] for row in cal])
    empirical = np.array([row["emp_retained"] for row in cal])
    sizes = np.array(
        [row["n"] for row in cal],
        dtype=float,
    )

    color = PR_CURVE_COLORS.get(selector_name, "teal")

    # Diagonal spans the same padded range as the axis limits below, so it
    # runs edge-to-edge instead of stopping at (0, 0)-(1, 1).
    diag_lo, diag_hi = -0.04, 1.08

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        [diag_lo, diag_hi],
        [diag_lo, diag_hi],
        linestyle="--",
        color="black",
        label="Ideal calibration",
    )

    ax.scatter(
        mean_scores,
        empirical,
        s=np.clip(sizes * 3, 20, 250),
        facecolor=color,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.8,
        label="Validation bins",
    )

    ax.plot(
        mean_scores,
        empirical,
        linestyle=":",
        color=color,
    )

    for x, y, n in zip(
        mean_scores,
        empirical,
        sizes.astype(int),
    ):
        ax.annotate(
            str(n),
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=10,
        )

    # Extra margin beyond [0, 1] so the per-bin count annotations (offset a
    # few points up-right of each marker) never sit outside the axes, even
    # for bins right at the (1, 1) corner.
    ax.set_xlim(diag_lo, diag_hi)
    ax.set_ylim(diag_lo, diag_hi)

    ax.set_xlabel(r"$s_{\mathrm{ret}}$", fontsize=15)
    ax.set_ylabel("Empirical retention fraction", fontsize=15)
    ax.set_title(f"Calibration: {selector_name} validation — O/H pooling")

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="minor", length=4)
    ax.tick_params(axis="both", labelsize=14)

    # The diagonal runs from the lower-left to the upper-right corner, so
    # the upper-left corner is the one guaranteed not to cross it (or the
    # calibration bins, which cluster near the diagonal for a well-
    # calibrated model).
    ax.legend(fontsize=14, loc="upper left")

    plt.tight_layout()

    path = f"motif_retention_calibration_{selector_name}_{OUTPUT_TAG}.{FIGURE_FORMAT}"

    plt.savefig(path, dpi=600)
    plt.close()

    print(f"Saved {path}")


def plot_precision_recall(
    y_true,
    p_retained,
    selector_name,
):
    y_true = np.asarray(y_true, dtype=int)
    p_retained = np.asarray(
        p_retained,
        dtype=float,
    )

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        print(
            f"Precision-recall plot skipped for {selector_name}: "
            "validation has fewer than two classes."
        )
        return

    precision, recall, _ = precision_recall_curve(
        y_true,
        p_retained,
    )

    ap = average_precision_score(
        y_true,
        p_retained,
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        recall,
        precision,
        linewidth=1.8,
        color=PR_CURVE_COLORS.get(selector_name, "teal"),
        label=f"AP={ap:.3f}",
    )

    # Extra headroom above the curve so the fixed top-right AP legend never
    # crosses the data (the curve itself never exceeds precision/recall=1).
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.14)

    ax.set_xlabel("Recall for retained", fontsize=15)
    ax.set_ylabel("Precision for retained", fontsize=15)

    ax.set_title(f"Precision-recall: {selector_name} validation — O/H pooling")

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major", length=8)
    ax.tick_params(which="minor", length=4)
    ax.tick_params(axis="both", labelsize=14)

    # Fixed position: top right, same spot on every selector's plot.
    ax.legend(fontsize=14, loc="upper right")

    plt.tight_layout()

    path = (
        f"motif_retention_precision_recall_"
        f"{selector_name}_{OUTPUT_TAG}.{FIGURE_FORMAT}"
    )

    plt.savefig(path, dpi=600)
    plt.close()

    print(f"Saved {path}")


def plot_head_comparison(rows):
    val_rows = [row for row in rows if row["split"] == "validation"]

    if not val_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="same score from both heads",
    )

    for domain in ["NP", "SLAB"]:
        subset = [row for row in val_rows if row["domain"] == domain]

        if not subset:
            continue

        x = [row["p_retained_np_head"] for row in subset]
        y = [row["p_retained_slab_head"] for row in subset]

        ax.scatter(
            x,
            y,
            s=18,
            alpha=0.65,
            label=f"{domain} validation",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("p(retained) from NP head")
    ax.set_ylabel("p(retained) from SLAB head")

    ax.set_title("NP head vs SLAB head — O/H pooling")

    ax.legend()

    plt.tight_layout()

    path = f"motif_retention_head_comparison_validation_{OUTPUT_TAG}.{FIGURE_FORMAT}"

    plt.savefig(path, dpi=600)
    plt.close()

    print(f"Saved {path}")


# ============================================================
# DATASET / MODEL
# ============================================================


def load_domain_dataset(domain_label):
    initial_file, final_file = DOMAIN_FILES[domain_label]

    print(f"\n{'=' * 72}")
    print(f"{domain_label}")
    print(f"Initial: {initial_file}")
    print(f"Final:   {final_file}")

    initial_frames = read(
        initial_file,
        index=":",
    )
    final_frames = read(
        final_file,
        index=":",
    )

    print(
        f"Loaded {len(initial_frames)} initial and "
        f"{len(final_frames)} final frames."
    )

    group_labels = build_group_labels(
        initial_frames,
        domain_label,
    )

    dataset, metadata = build_dataset(
        initial_frames,
        final_frames,
        group_labels,
        domain_label,
    )

    return dataset, metadata


def predict_one(model, data):
    """Predict one graph with both heads and return the domain-routed score."""
    data = data.to(DEVICE)

    batch_vec = torch.zeros(
        data.x.shape[0],
        dtype=torch.long,
        device=DEVICE,
    )

    embedding = model.encode(
        data.x,
        data.edge_index,
        data.edge_attr,
        batch_vec,
    )

    np_logit = model.np_retention_head(embedding).squeeze(-1)

    slab_logit = model.slab_retention_head(embedding).squeeze(-1)

    p_np = float(torch.sigmoid(np_logit).item())

    p_slab = float(torch.sigmoid(slab_logit).item())

    domain_id = int(data.domain_id.view(-1).item())

    if domain_id == DOMAIN_TO_ID["NP"]:
        p_routed = p_np

    elif domain_id == DOMAIN_TO_ID["SLAB"]:
        p_routed = p_slab

    else:
        raise ValueError(f"Unknown domain_id={domain_id}")

    return p_routed, p_np, p_slab


def split_name(
    index,
    train_idx,
    val_idx,
):
    if index in train_idx:
        return "train"

    if index in val_idx:
        return "validation"

    return "unknown"


def validate_checkpoint(checkpoint):
    """Fail early when the checkpoint predates the current data contract."""
    expected_labels = {"0": "migrated", "1": "retained_exact"}
    if checkpoint.get("binary_label_definition") != expected_labels:
        raise ValueError(
            "Checkpoint binary labels do not match the current retention model. "
            "Retrain with train_retention_classifier.py before analysis."
        )

    settings = checkpoint.get("settings", {})
    required_settings = {
        "OH_BOND_CUTOFF": OH_BOND_CUTOFF,
        "MIN_OH_SEPARATION": MIN_OH_SEPARATION,
        "USE_HEADER_INITIAL_SITE": USE_HEADER_INITIAL_SITE,
    }
    for key, expected in required_settings.items():
        if key not in settings:
            raise ValueError(
                f"Checkpoint is missing setting {key!r}; retrain with the "
                "current retention-training script."
            )
        actual = settings[key]
        if isinstance(expected, float):
            matches = np.isclose(float(actual), expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Checkpoint setting {key}={actual!r} does not match the "
                f"current value {expected!r}."
            )

    for domain_label in ("NP", "SLAB"):
        if domain_label not in checkpoint.get("domain_splits", {}):
            raise ValueError(
                f"Checkpoint is missing the {domain_label} train/validation split."
            )


# ============================================================
# REPORTING
# ============================================================


def report_per_domain_motif(rows):
    print("\n=== PER DOMAIN / INITIAL MOTIF ===")

    print(
        f"  {'domain':6s} {'motif':6s} {'n':>5s} "
        f"{'retained':>9s} {'ret_frac':>9s} "
        f"{'mean_p':>9s} {'acc@0.50':>10s}"
    )

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["domain"],
                row["motif_initial"],
            )
        ].append(row)

    for domain, motif in sorted(grouped):
        subset = grouped[(domain, motif)]

        y = np.array(
            [row["true_retained"] for row in subset],
            dtype=int,
        )

        p = np.array(
            [row["p_retained"] for row in subset],
            dtype=float,
        )

        pred = (p >= 0.50).astype(int)

        print(
            f"  {domain:6s} {motif:6s} "
            f"{len(subset):5d} "
            f"{int(y.sum()):9d} "
            f"{y.mean():9.4f} "
            f"{p.mean():9.4f} "
            f"{accuracy_score(y, pred):10.4f}"
        )


def report_per_domain_motif_validation(rows):
    print("\n=== VALIDATION: PER DOMAIN / INITIAL MOTIF ===")

    precision_label = f"prec@{MOTIF_REPORT_THRESHOLD:.2f}"
    recall_label = f"rec@{MOTIF_REPORT_THRESHOLD:.2f}"
    print(
        f"  {'domain':6s} {'motif':6s} {'n':>5s} "
        f"{'retained':>9s} {'ret_frac':>9s} "
        f"{'mean_p':>9s} {precision_label:>9s} {recall_label:>9s}"
    )

    grouped = defaultdict(list)

    for row in rows:
        if row["split"] == "validation":
            grouped[
                (
                    row["domain"],
                    row["motif_initial"],
                )
            ].append(row)

    for domain, motif in sorted(grouped):
        subset = grouped[(domain, motif)]

        y = np.array(
            [row["true_retained"] for row in subset],
            dtype=int,
        )

        p = np.array(
            [row["p_retained"] for row in subset],
            dtype=float,
        )

        m = binary_metrics(
            y,
            p,
            threshold=MOTIF_REPORT_THRESHOLD,
        )

        print(
            f"  {domain:6s} {motif:6s} "
            f"{len(subset):5d} "
            f"{int(y.sum()):9d} "
            f"{y.mean():9.4f} "
            f"{p.mean():9.4f} "
            f"{m['precision']:9.4f} "
            f"{m['recall']:9.4f}"
        )


def report_head_gap(rows):
    print("\n=== NP HEAD vs SLAB HEAD ===")

    print(
        "Positive gap = SLAB head gives a larger p(retained) "
        "for the same encoded graph."
    )

    print(
        f"  {'domain':6s} {'motif':6s} {'n':>5s} "
        f"{'NP head':>9s} {'SLAB head':>10s} {'gap':>9s}"
    )

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["domain"],
                row["motif_initial"],
            )
        ].append(row)

    for domain, motif in sorted(grouped):
        subset = grouped[(domain, motif)]

        p_np = np.array(
            [row["p_retained_np_head"] for row in subset],
            dtype=float,
        )

        p_slab = np.array(
            [row["p_retained_slab_head"] for row in subset],
            dtype=float,
        )

        print(
            f"  {domain:6s} {motif:6s} "
            f"{len(subset):5d} "
            f"{p_np.mean():9.4f} "
            f"{p_slab.mean():10.4f} "
            f"{(p_slab - p_np).mean():9.4f}"
        )


# ============================================================
# MAIN
# ============================================================


def main():
    print(f"Device: {DEVICE}")
    print(f"Training module: {TRAINING_MODULE}")
    print(f"Checkpoint: {MODEL_PATH}")
    print("Pooling: O-focused + H-focused weighted pooling")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )
    validate_checkpoint(checkpoint)

    print("\nCheckpoint info:")
    print(f"  task: {checkpoint.get('task', 'n/a')}")
    print(f"  architecture: {checkpoint.get('architecture', 'n/a')}")
    print(f"  split_mode: {checkpoint.get('split_mode', 'n/a')}")
    print(f"  hidden_dim: {checkpoint.get('hidden_dim', 'n/a')}")
    print(
        "  early_stopping_metric: " f"{checkpoint.get('early_stopping_metric', 'n/a')}"
    )
    print(
        f"  best_domain_macro_value: "
        f"{checkpoint.get('best_domain_macro_value', 'n/a')}"
    )

    settings = checkpoint.get(
        "settings",
        {},
    )

    if settings:
        print(f"  training_seed: {settings.get('TRAINING_SEED', 'n/a')}")
        print(f"  split_seed: {settings.get('SPLIT_SEED', 'n/a')}")
        print(f"  fixed_split: {settings.get('USE_FIXED_SPLIT', 'n/a')}")
        print("  initial_site_source: " f"{settings.get('INITIAL_SITE_SOURCE', 'n/a')}")
        print(
            "  adsorption_state_filter: exactly one O-H below "
            f"{settings['OH_BOND_CUTOFF']:.2f} Å; minimum separation "
            f"{settings['MIN_OH_SEPARATION']:.2f} Å"
        )

    model = MotifRetentionTwoHeadGNN(
        node_dim=checkpoint["node_dim"],
        edge_dim=checkpoint["edge_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []

    for domain_label in ["NP", "SLAB"]:
        dataset, metadata = load_domain_dataset(domain_label)

        split_info = checkpoint["domain_splits"][domain_label]

        train_idx = set(split_info["train_idx"])
        val_idx = set(split_info["val_idx"])

        expected_idx = set(range(len(dataset)))
        if train_idx & val_idx:
            raise ValueError(f"{domain_label}: train/validation indices overlap.")
        if train_idx | val_idx != expected_idx:
            raise ValueError(
                f"{domain_label}: checkpoint split does not cover the rebuilt "
                "dataset exactly. Retrain after changing the dataset/filter."
            )

        print(
            f"{domain_label} split: "
            f"train={len(train_idx)} "
            f"validation={len(val_idx)}"
        )

        with torch.no_grad():
            for domain_index, (
                data,
                meta,
            ) in enumerate(zip(dataset, metadata)):
                p_routed, p_np, p_slab = predict_one(
                    model,
                    data,
                )

                true_retained = int(data.y_stable.item())

                pred_retained_050 = int(p_routed >= 0.50)

                rows.append(
                    {
                        "domain": domain_label,
                        "domain_index": int(domain_index),
                        "frame_idx": int(meta["frame_idx"]),
                        "split": split_name(
                            domain_index,
                            train_idx,
                            val_idx,
                        ),
                        "group": meta["group"],
                        "config_label": meta.get(
                            "config_label",
                            "",
                        ),
                        "n_ni": int(meta["n_ni"]),
                        "motif_initial": meta["motif_initial"],
                        "initial_site_source": meta["initial_site_source"],
                        "motif_final": meta["motif_final"],
                        "same_motif": bool(meta["same_motif"]),
                        "same_o_anchor": bool(meta["same_o_anchor"]),
                        "same_h_anchor": bool(meta["same_h_anchor"]),
                        "true_retained": (true_retained),
                        "pred_retained_at_0.50": (pred_retained_050),
                        "correct_at_0.50": int(pred_retained_050 == true_retained),
                        "p_retained": float(p_routed),
                        "p_retained_np_head": float(p_np),
                        "p_retained_slab_head": float(p_slab),
                        "head_gap_slab_minus_np": float(p_slab - p_np),
                        "o_anchor_initial": "-".join(
                            map(
                                str,
                                meta["o_anchor_initial"],
                            )
                        ),
                        "h_anchor_initial": "-".join(
                            map(
                                str,
                                meta["h_anchor_initial"],
                            )
                        ),
                        "o_anchor_final": "-".join(
                            map(
                                str,
                                meta["o_anchor_final"],
                            )
                        ),
                        "h_anchor_final": "-".join(
                            map(
                                str,
                                meta["h_anchor_final"],
                            )
                        ),
                    }
                )

    if not rows:
        raise RuntimeError("No rows generated.")

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------
    with open(
        OUT_CSV,
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved predictions to: {OUT_CSV}")

    # --------------------------------------------------------
    # Combined arrays
    # --------------------------------------------------------
    y = np.array(
        [row["true_retained"] for row in rows],
        dtype=int,
    )

    p = np.array(
        [row["p_retained"] for row in rows],
        dtype=float,
    )

    val_mask = np.array([row["split"] == "validation" for row in rows])

    np_mask = np.array([row["domain"] == "NP" for row in rows])

    slab_mask = np.array([row["domain"] == "SLAB" for row in rows])

    # --------------------------------------------------------
    # Threshold reports
    # --------------------------------------------------------
    print("\n=== ALL DATA ===")

    for threshold in THRESHOLDS:
        print(f"\nThreshold = {threshold:.2f}")

        print_binary_report(
            "ALL",
            y,
            p,
            threshold,
        )

        print_binary_report(
            "NP",
            y[np_mask],
            p[np_mask],
            threshold,
        )

        print_binary_report(
            "SLAB",
            y[slab_mask],
            p[slab_mask],
            threshold,
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    print("\n=== VALIDATION ONLY ===")

    selectors = {
        "ALL": val_mask,
        "NP": (val_mask & np_mask),
        "SLAB": (val_mask & slab_mask),
    }

    for threshold in THRESHOLDS:
        print(f"\nThreshold = {threshold:.2f}")

        for name, mask in selectors.items():
            print_binary_report(
                name,
                y[mask],
                p[mask],
                threshold,
            )

    # --------------------------------------------------------
    # Train vs validation
    # --------------------------------------------------------
    print("\n=== TRAIN vs VALIDATION @ 0.50 ===")

    for split in [
        "train",
        "validation",
    ]:
        split_mask = np.array([row["split"] == split for row in rows])

        print(f"\n{split.upper()}")

        print_binary_report(
            "ALL",
            y[split_mask],
            p[split_mask],
            0.50,
        )

        print_binary_report(
            "NP",
            y[split_mask & np_mask],
            p[split_mask & np_mask],
            0.50,
        )

        print_binary_report(
            "SLAB",
            y[split_mask & slab_mask],
            p[split_mask & slab_mask],
            0.50,
        )

    # --------------------------------------------------------
    # Detailed diagnostics
    # --------------------------------------------------------
    report_per_domain_motif(rows)

    report_per_domain_motif_validation(rows)

    report_head_gap(rows)

    # --------------------------------------------------------
    # Calibration tables
    # --------------------------------------------------------
    print("\n=== CALIBRATION-LIKE BINNING: VALIDATION ===")

    for name, mask in selectors.items():
        print(f"\n{name}")

        print(
            f"  {'bin':>14s} "
            f"{'n':>5s} "
            f"{'mean_score':>11s} "
            f"{'emp_retained':>13s}"
        )

        for row in calibration_bins(
            y[mask],
            p[mask],
            n_bins=10,
        ):
            print(
                f"  "
                f"[{row['lo']:.1f},"
                f"{row['hi']:.1f}): "
                f"{row['n']:5d} "
                f"{row['mean_score']:11.4f} "
                f"{row['emp_retained']:13.4f}"
            )

    # --------------------------------------------------------
    # Threshold scan files
    # --------------------------------------------------------
    for name, mask in selectors.items():
        path = THRESHOLD_SCAN_FILES[name]

        save_threshold_scan(
            y[mask],
            p[mask],
            path,
        )

        print(f"Saved {path}")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    for name in [
        "ALL",
        "NP",
        "SLAB",
    ]:
        plot_retention_score(
            rows,
            name,
        )

    for name, mask in selectors.items():
        plot_calibration(
            y[mask],
            p[mask],
            name,
        )

        plot_precision_recall(
            y[mask],
            p[mask],
            name,
        )

    plot_head_comparison(rows)


if __name__ == "__main__":
    main()
