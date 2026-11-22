#!/usr/bin/env python3
"""Create the SI baseline-comparison figures and the external-test evaluation.

Run from the ``baselines`` directory (``python make_baseline_figures.py``).

Figure 1 is a 2 x 3 internal-validation parity grid for Ridge, XGBoost,
and M6-MACE (Figure S8). Figure 2 reports absolute- and relative-energy MAEs
on the five external test systems (Figure S9). The external descriptors are
computed on the MACE-relaxed bare test structures from the final DFT motifs
and anchor sets, which are parsed with ``test_systems/analysis_energy_test_systems.py``.

Relative energies use the manuscript convention: for each system, the DFT
minimum selects the reference configuration, and both DFT and predicted
energies are shifted by their respective values for that same configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-baseline-paper-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from periodic_site import site_gcn_pbc, site_centroid_distance_pbc


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
RESULTS_DIR = HERE / "reference_results"
OUTPUT_DIR = HERE / "reproduced_results"
M6_INTERNAL_ERRORS = HERE / "datasets" / "M6_energy_model_predictions.csv"
MODEL_VALIDATION_ROOT = REPOSITORY_ROOT / "test_systems"
MODEL_VALIDATION_SCRIPT = MODEL_VALIDATION_ROOT / "analysis_energy_test_systems.py"
MACE_BARE_DIR = REPOSITORY_ROOT / "ML_relaxed_bare_geom" / "MACE" / "testing"

MODEL_ORDER = ("Ridge", "XGBoost", "M6-MACE")
MODEL_COLORS = {
    "Ridge": "#7F7F7F",
    "XGBoost": "#DD8452",
    "M6-MACE": "#4C72B0",
}
DOMAIN_STYLE = {
    "NP": {
        "label": "Nanoclusters",
        "marker": "o",
        "face": mcolors.to_rgba("royalblue", alpha=0.3),
        "edge": "navy",
    },
    "SLAB": {
        "label": "Surfaces",
        "marker": "o",
        "face": mcolors.to_rgba("firebrick", alpha=0.3),
        "edge": "darkred",
    },
}
GCN_CN_MAX = {"T": 12.0, "B": 18.0, "H": 22.0, "F": 26.0}
NI_NEIGHBOR_CUTOFF = 3.00


@dataclass(frozen=True)
class ExternalSystem:
    folder: str
    label: str
    plot_label: str
    bare_filename: str


EXTERNAL_SYSTEMS = (
    ExternalSystem("Ni55", "Ni55", r"Ni$_{55}$", "Ni55_MACE_opt.xyz"),
    ExternalSystem("Ni91", "Ni91", r"Ni$_{91}$", "Ni91_MACE_opt.xyz"),
    ExternalSystem("Ni130", "Ni130", r"Ni$_{130}$", "Ni130_MACE_opt.xyz"),
    ExternalSystem(
        "Ni111_4x3", "Ni(111)", "Ni(111)-4x3",
        "Ni111_4x3_MACE_opt.extxyz",
    ),
    ExternalSystem(
        "Ni100_4x4", "Ni(100)", "Ni(100)-4x4",
        "Ni100_4x4_MACE_opt.extxyz",
    ),
)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    errors = pred - true
    return {
        "n": int(len(true)),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "R2": float(r2_score(true, pred)) if len(true) >= 2 else float("nan"),
    }


def add_relative_energies(frame: pd.DataFrame) -> pd.DataFrame:
    """Shift each model to the configuration selected by the DFT minimum."""
    result = frame.copy()
    result["DFT_relative_eV"] = np.nan
    result["pred_relative_eV"] = np.nan
    grouping_columns = ["system", "model"] if "model" in result.columns else ["system"]
    for _, indices in result.groupby(grouping_columns, sort=False).groups.items():
        group = result.loc[indices]
        reference_index = group["DFT_Eads_eV"].idxmin()
        result.loc[indices, "DFT_relative_eV"] = (
            group["DFT_Eads_eV"] - result.loc[reference_index, "DFT_Eads_eV"]
        )
        result.loc[indices, "pred_relative_eV"] = (
            group["pred_Eads_eV"] - result.loc[reference_index, "pred_Eads_eV"]
        )
    return result


def verify_internal_inputs(models: dict[str, pd.DataFrame]) -> None:
    """Confirm that each model is evaluated on the same validation targets."""
    reference = models["Ridge"]
    reference_counts = reference.groupby(["domain", "system"]).size().sort_index()
    for model_name, frame in models.items():
        counts = frame.groupby(["domain", "system"]).size().sort_index()
        if not counts.equals(reference_counts):
            raise ValueError(f"{model_name}: validation group counts differ from Ridge")
        for system, reference_group in reference.groupby("system"):
            model_values = np.sort(
                frame.loc[frame["system"] == system, "DFT_Eads_eV"].to_numpy(float)
            )
            reference_values = np.sort(reference_group["DFT_Eads_eV"].to_numpy(float))
            # The two saved files differ only by float32/float64 serialization
            # (observed maximum discrepancy below 6e-8 eV).
            if not np.allclose(model_values, reference_values, atol=1e-6, rtol=0.0):
                raise ValueError(f"{model_name}: DFT targets differ for {system}")


def load_internal_models(
    results_dir: Path, m6_errors_path: Path
) -> dict[str, pd.DataFrame]:
    baseline = pd.read_csv(results_dir / "baseline_validation_predictions.csv")
    m6 = pd.read_csv(m6_errors_path)
    m6 = m6[m6["split"] == "validation"].copy()

    common_baseline = baseline[["domain", "system", "Eads_DFT_eV"]].rename(
        columns={"Eads_DFT_eV": "DFT_Eads_eV"}
    )
    ridge = common_baseline.assign(pred_Eads_eV=baseline["Ridge_pred_eV"].to_numpy())
    xgboost = common_baseline.assign(
        pred_Eads_eV=baseline["Nonlinear_pred_eV"].to_numpy()
    )
    m6 = m6.rename(columns={
        "group": "system",
        "true_energy": "DFT_Eads_eV",
        "pred_energy": "pred_Eads_eV",
    })[["domain", "system", "DFT_Eads_eV", "pred_Eads_eV"]]

    models = {"Ridge": ridge, "XGBoost": xgboost, "M6-MACE": m6}
    verify_internal_inputs(models)
    return models


def common_parity_limits(
    frames: list[pd.DataFrame], true_column: str, pred_column: str
) -> tuple[float, float]:
    values = np.concatenate([
        frame[[true_column, pred_column]].to_numpy(float).ravel()
        for frame in frames
    ])
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    padding = 0.055 * max(high - low, 0.1)
    return low - padding, high + padding


def scatter_domains(
    axis: plt.Axes,
    frame: pd.DataFrame,
    true_column: str,
    pred_column: str,
) -> None:
    for domain in ("NP", "SLAB"):
        subset = frame[frame["domain"] == domain]
        style = DOMAIN_STYLE[domain]
        axis.scatter(
            subset[true_column],
            subset[pred_column],
            s=70,
            marker=style["marker"],
            facecolor=style["face"],
            edgecolor=style["edge"],
            linewidth=1.5,
            zorder=3,
        )


def style_parity_axis(
    axis: plt.Axes,
    limits: tuple[float, float],
    x_label: str,
    y_label: str | None,
) -> None:
    axis.plot(limits, limits, "k--", linewidth=1.2, zorder=2)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(x_label, fontsize=20)
    if y_label is not None:
        axis.set_ylabel(y_label, fontsize=20)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(which="major", labelsize=16, length=6, width=1.0)
    axis.tick_params(which="minor", length=3.2, width=0.8)
    axis.grid(True, alpha=0.20, linewidth=0.7, zorder=1)


def print_internal_metrics(models: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 77)
    print("INTERNAL VALIDATION")
    print("=" * 77)
    print("Relative energies use the DFT-minimum configuration as the common reference.")
    print(f"{'representation':<12} {'model':<10} {'n':>5} {'MAE (eV)':>11} "
          f"{'RMSE (eV)':>12} {'R2':>9}")
    print("-" * 77)
    for representation in ("absolute", "relative"):
        for model_name in MODEL_ORDER:
            frame = models[model_name]
            if representation == "relative":
                frame = add_relative_energies(frame)
                true_column, pred_column = "DFT_relative_eV", "pred_relative_eV"
            else:
                true_column, pred_column = "DFT_Eads_eV", "pred_Eads_eV"
            metrics = regression_metrics(frame[true_column], frame[pred_column])
            print(
                f"{representation:<12} {model_name:<10} {metrics['n']:>5d} "
                f"{metrics['MAE']:>11.3f} {metrics['RMSE']:>12.3f} "
                f"{metrics['R2']:>9.3f}"
            )


def plot_internal_validation(
    models: dict[str, pd.DataFrame], output_dir: Path
) -> Path:
    relative_models = {
        model_name: add_relative_energies(frame)
        for model_name, frame in models.items()
    }
    absolute_limits = common_parity_limits(
        list(models.values()), "DFT_Eads_eV", "pred_Eads_eV"
    )
    relative_limits = common_parity_limits(
        list(relative_models.values()), "DFT_relative_eV", "pred_relative_eV"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 10.1))
    for column, model_name in enumerate(MODEL_ORDER):
        absolute = models[model_name]
        relative = relative_models[model_name]
        scatter_domains(axes[0, column], absolute, "DFT_Eads_eV", "pred_Eads_eV")
        scatter_domains(
            axes[1, column], relative, "DFT_relative_eV", "pred_relative_eV"
        )
        style_parity_axis(
            axes[0, column],
            absolute_limits,
            r"DFT $\mathrm{E}_{\mathrm{ads}}$ (eV)",
            r"Predicted $\mathrm{E}_{\mathrm{ads}}$ (eV)" if column == 0 else None,
        )
        style_parity_axis(
            axes[1, column],
            relative_limits,
            r"DFT $\Delta \mathrm{E}_{\mathrm{ads}}$ (eV)",
            r"Predicted $\Delta \mathrm{E}_{\mathrm{ads}}$ (eV)"
            if column == 0 else None,
        )
        axes[0, column].set_title(model_name, fontsize=21, pad=10)

    handles = [
        Line2D(
            [0], [0], linestyle="none", marker=DOMAIN_STYLE[domain]["marker"],
            markersize=11,
            markerfacecolor=DOMAIN_STYLE[domain]["face"],
            markeredgecolor=DOMAIN_STYLE[domain]["edge"],
            markeredgewidth=1.5,
            label=DOMAIN_STYLE[domain]["label"],
        )
        for domain in ("NP", "SLAB")
    ]
    axes[0, 0].legend(
        handles=handles,
        loc="upper left",
        fontsize=16,
        frameon=True,
        framealpha=0.90,
        facecolor="none",
        edgecolor="gray",
        borderpad=0.35,
        labelspacing=0.45,
        handletextpad=0.55,
    )
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.075, top=0.955,
        wspace=0.23, hspace=0.26,
    )

    pdf_path = output_dir / "baseline_internal_validation.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("model_validation_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_coordination(atoms) -> tuple[list[list[tuple[int, tuple[int, int, int]]]], np.ndarray]:
    i_indices, j_indices, shifts = neighbor_list(
        "ijS", atoms, NI_NEIGHBOR_CUTOFF, self_interaction=False
    )
    neighbors: list[list[tuple[int, tuple[int, int, int]]]] = [
        [] for _ in range(len(atoms))
    ]
    for i_index, j_index, shift in zip(i_indices, j_indices, shifts):
        neighbors[int(i_index)].append(
            (int(j_index), tuple(int(value) for value in shift))
        )
    coordination = np.asarray([len(items) for items in neighbors], dtype=float)
    return neighbors, coordination


def parse_anchor_string(value: object) -> list[int]:
    return [int(token) for token in str(value).split(",") if token.strip()]


def ridge_design_row(descriptors: dict[str, object], payload: dict) -> np.ndarray:
    values = np.asarray([
        descriptors["GCN_O"], descriptors["GCN_H"], descriptors["d_OH_A"]
    ], dtype=float)
    standardized = (
        values - np.asarray(payload["scaler_mean"], dtype=float)
    ) / np.asarray(payload["scaler_scale"], dtype=float)
    z_o, z_h, z_distance = standardized
    one_hot = np.asarray([
        float(descriptors["motif"] == motif) for motif in payload["motifs"]
    ])
    return np.concatenate((
        np.asarray([z_o, z_h, z_distance, z_o * z_h]),
        one_hot,
        one_hot * z_o,
    ))


def xgboost_design_row(descriptors: dict[str, object], payload: dict) -> np.ndarray:
    one_hot = np.asarray([
        float(descriptors["motif"] == motif) for motif in payload["motifs"]
    ])
    return np.concatenate((
        np.asarray([
            descriptors["GCN_O"], descriptors["GCN_H"], descriptors["d_OH_A"]
        ], dtype=float),
        one_hot,
    ))


def predict_baselines(
    descriptors: dict[str, object],
    ridge_payload: dict,
    xgboost_model: XGBRegressor,
    xgboost_preprocessing: dict,
) -> tuple[float, float]:
    ridge_features = ridge_design_row(descriptors, ridge_payload)
    ridge_coefficients = np.asarray(ridge_payload["coefficients"], dtype=float)
    ridge_prediction = float(
        ridge_payload["intercept"] + ridge_features @ ridge_coefficients
    )
    xgboost_features = xgboost_design_row(
        descriptors, xgboost_preprocessing
    ).reshape(1, -1)
    xgboost_prediction = float(xgboost_model.predict(xgboost_features)[0])
    return ridge_prediction, xgboost_prediction


def descriptor_is_supported(descriptors: dict[str, object], payload: dict) -> bool:
    if descriptors["motif"] not in payload["motifs"]:
        return False
    for name, bounds in payload["training_descriptor_ranges"].items():
        value = float(descriptors[name])
        if value < float(bounds["min"]) or value > float(bounds["max"]):
            return False
    return True


def build_external_predictions(
    results_dir: Path,
    validation_root: Path,
    validation_script: Path,
    mace_dir: Path,
) -> pd.DataFrame:
    helpers = load_module(validation_script)
    helper_specs = {spec.folder: spec for spec in helpers.SYSTEMS}

    ridge_payload = json.loads(
        (results_dir / "ridge_baseline_portable.json").read_text()
    )
    xgboost_preprocessing = json.loads(
        (results_dir / "nonlinear_baseline_preprocessing.json").read_text()
    )
    xgboost_model = XGBRegressor()
    xgboost_model.load_model(results_dir / "xgboost_baseline_native.json")

    rows: list[dict[str, object]] = []
    for system in EXTERNAL_SYSTEMS:
        frame, errors = helpers.process_system(
            validation_root, helper_specs[system.folder]
        )
        frame = frame.dropna(subset=["DFT_Eads_eV", "M6_Eads_eV"]).copy()
        bare = read(mace_dir / system.bare_filename)
        if set(bare.get_chemical_symbols()) != {"Ni"}:
            raise ValueError(f"{system.label}: bare reference is not Ni-only")
        neighbors, coordination = build_coordination(bare)
        supported_count = 0

        for _, row in frame.iterrows():
            o_anchors = parse_anchor_string(row["final_o_anchor_idx"])
            h_anchors = parse_anchor_string(row["final_h_anchor_idx"])
            motif = str(row["final_motif"])
            descriptors = {
                "motif": motif,
                "GCN_O": site_gcn_pbc(
                    o_anchors, motif[0], neighbors, coordination,
                    bare.get_positions(), bare.get_cell(), bare.get_pbc(),
                )[0],
                "GCN_H": site_gcn_pbc(
                    h_anchors, motif[1], neighbors, coordination,
                    bare.get_positions(), bare.get_cell(), bare.get_pbc(),
                )[0],
                "d_OH_A": site_centroid_distance_pbc(
                    bare.get_positions(), o_anchors, h_anchors,
                    bare.get_cell(), bare.get_pbc(),
                ),
            }
            ridge_prediction, xgboost_prediction = predict_baselines(
                descriptors, ridge_payload, xgboost_model, xgboost_preprocessing
            )
            supported = descriptor_is_supported(descriptors, ridge_payload)
            supported_count += int(supported)
            common = {
                "system": system.label,
                "folder": system.folder,
                "config": int(row["config"]),
                "domain": row["system_type"],
                "DFT_Eads_eV": float(row["DFT_Eads_eV"]),
                "motif": motif,
                "GCN_O": descriptors["GCN_O"],
                "GCN_H": descriptors["GCN_H"],
                "d_OH_A": descriptors["d_OH_A"],
                "inside_training_descriptor_ranges": supported,
            }
            rows.extend([
                {**common, "model": "Ridge", "pred_Eads_eV": ridge_prediction},
                {**common, "model": "XGBoost", "pred_Eads_eV": xgboost_prediction},
                {**common, "model": "M6-MACE", "pred_Eads_eV": float(row["M6_Eads_eV"])},
            ])

        print(
            f"{system.label}: {len(frame)} common DFT/model configurations; "
            f"{supported_count}/{len(frame)} inside all baseline descriptor ranges; "
            f"{len(errors)} relaxed structures skipped by the validation parser"
        )

    return pd.DataFrame(rows)


def transfer_metric_table(external: pd.DataFrame) -> pd.DataFrame:
    relative = add_relative_energies(external)
    rows = []
    for system in EXTERNAL_SYSTEMS:
        for model_name in MODEL_ORDER:
            absolute_subset = external[
                (external["system"] == system.label)
                & (external["model"] == model_name)
            ]
            relative_subset = relative[
                (relative["system"] == system.label)
                & (relative["model"] == model_name)
            ]
            absolute_metrics = regression_metrics(
                absolute_subset["DFT_Eads_eV"], absolute_subset["pred_Eads_eV"]
            )
            relative_metrics = regression_metrics(
                relative_subset["DFT_relative_eV"],
                relative_subset["pred_relative_eV"],
            )
            rows.append({
                "system": system.label,
                "model": model_name,
                "n": absolute_metrics["n"],
                "absolute_MAE": absolute_metrics["MAE"],
                "absolute_RMSE": absolute_metrics["RMSE"],
                "absolute_R2": absolute_metrics["R2"],
                "relative_MAE": relative_metrics["MAE"],
                "relative_RMSE": relative_metrics["RMSE"],
                "relative_R2": relative_metrics["R2"],
            })
    return pd.DataFrame(rows)


def print_transfer_metrics(metrics: pd.DataFrame) -> None:
    print("\n" + "=" * 108)
    print("TRANSFERABILITY TO UNSEEN SYSTEMS")
    print("=" * 108)
    print("Relative energies use the DFT-minimum configuration as the common reference.")
    print(
        f"{'system':<10} {'model':<10} {'n':>4}  "
        f"{'abs MAE':>8} {'abs RMSE':>9} {'abs R2':>8}  "
        f"{'rel MAE':>8} {'rel RMSE':>9} {'rel R2':>8}"
    )
    print("-" * 108)
    for _, row in metrics.iterrows():
        print(
            f"{row['system']:<10} {row['model']:<10} {int(row['n']):>4d}  "
            f"{row['absolute_MAE']:>8.3f} {row['absolute_RMSE']:>9.3f} "
            f"{row['absolute_R2']:>8.3f}  {row['relative_MAE']:>8.3f} "
            f"{row['relative_RMSE']:>9.3f} {row['relative_R2']:>8.3f}"
        )


def plot_transferability(
    metrics: pd.DataFrame, output_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), sharey=True)
    x_positions = np.arange(len(EXTERNAL_SYSTEMS), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    maximum_mae = float(
        metrics[["absolute_MAE", "relative_MAE"]].to_numpy().max()
    )
    upper_limit = max(0.10, np.ceil(maximum_mae * 12.0) / 10.0)

    for axis, metric_column, y_label in (
        (axes[0], "absolute_MAE", r"Absolute $\mathrm{E}_{\mathrm{ads}}$ MAE (eV)"),
        (axes[1], "relative_MAE", r"Relative $\Delta \mathrm{E}_{\mathrm{ads}}$ MAE (eV)"),
    ):
        for offset, model_name in zip(offsets, MODEL_ORDER):
            values = []
            for system in EXTERNAL_SYSTEMS:
                value = metrics.loc[
                    (metrics["system"] == system.label)
                    & (metrics["model"] == model_name),
                    metric_column,
                ]
                values.append(float(value.iloc[0]))
            axis.bar(
                x_positions + offset,
                values,
                width=width,
                color=MODEL_COLORS[model_name],
                edgecolor="black",
                linewidth=0.75,
                label=model_name,
                zorder=3,
            )
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [system.plot_label for system in EXTERNAL_SYSTEMS], fontsize=15
        )
        for tick_label in axis.get_xticklabels():
            tick_label.set_color("black")
        axis.set_ylabel(y_label, fontsize=18)
        axis.set_ylim(0.0, upper_limit)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.tick_params(axis="y", which="major", labelsize=15, length=6)
        axis.tick_params(axis="y", which="minor", length=3)
        axis.tick_params(axis="x", length=0, pad=8)
        axis.grid(axis="y", alpha=0.23, linewidth=0.8, zorder=1)
        axis.set_axisbelow(True)

    handles = [
        Patch(
            facecolor=MODEL_COLORS[model_name], edgecolor="black",
            linewidth=0.75, label=model_name,
        )
        for model_name in MODEL_ORDER
    ]
    axes[0].legend(
        handles=handles,
        loc="upper left",
        fontsize=16,
        frameon=True,
        framealpha=0.90,
        facecolor="none",
        edgecolor="gray",
        borderpad=0.45,
        labelspacing=0.45,
        handletextpad=0.55,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.17, top=0.955, wspace=0.22
    )

    pdf_path = output_dir / "baseline_transferability_mae.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Directory with the trained baseline models and validation predictions.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--m6-internal-errors", type=Path, default=M6_INTERNAL_ERRORS)
    parser.add_argument("--validation-root", type=Path, default=MODEL_VALIDATION_ROOT)
    parser.add_argument(
        "--validation-script", type=Path, default=MODEL_VALIDATION_SCRIPT
    )
    parser.add_argument("--mace-bare-dir", type=Path, default=MACE_BARE_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    internal_models = load_internal_models(
        args.results_dir, args.m6_internal_errors
    )
    print_internal_metrics(internal_models)
    internal_pdf = plot_internal_validation(
        internal_models, args.output_dir
    )

    print("\nExternal-system parsing and baseline descriptor coverage:")
    external = build_external_predictions(
        args.results_dir,
        args.validation_root,
        args.validation_script,
        args.mace_bare_dir,
    )
    transfer_metrics = transfer_metric_table(external)
    print_transfer_metrics(transfer_metrics)
    transfer_pdf = plot_transferability(
        transfer_metrics, args.output_dir
    )
    external.to_csv(args.output_dir / "baseline_external_predictions.csv", index=False)
    transfer_metrics.to_csv(args.output_dir / "baseline_external_metrics.csv", index=False)

    print("\nWrote figures:")
    for path in (internal_pdf, transfer_pdf):
        print(f"  {path}")


if __name__ == "__main__":
    main()
