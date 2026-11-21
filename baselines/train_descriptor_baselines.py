#!/usr/bin/env python3
"""Tune and evaluate Ridge and XGBoost descriptor baselines against M6-MACE.

Run this script from the ``baselines`` directory. By default it reads the
processed baseline dataset and the saved M6 predictions from ``datasets/`` and
writes a fresh set of models, predictions, metrics, and plots to
``reproduced_results/``. All default inputs are distributed with this folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-baseline-cache")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


SEED = 32
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "datasets" / "baseline_dataset.csv"
DEFAULT_M6_PREDICTIONS = HERE / "datasets" / "M6_energy_model_predictions.csv"


class RidgeFeatureMap:
    """Physically motivated Ridge features from the baseline specification."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.motifs: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "RidgeFeatureMap":
        self.scaler.fit(frame[["GCN_O", "GCN_H", "d_OH_A"]].to_numpy(float))
        self.motifs = sorted(frame["motif"].astype(str).unique())
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        values = self.scaler.transform(frame[["GCN_O", "GCN_H", "d_OH_A"]].to_numpy(float))
        z_o, z_h, z_d = values.T
        one_hot = np.column_stack(
            [(frame["motif"].astype(str).to_numpy() == motif).astype(float) for motif in self.motifs]
        )
        matrix = np.column_stack([z_o, z_h, z_d, z_o * z_h, one_hot, one_hot * z_o[:, None]])
        names = (
            ["z_GCN_O", "z_GCN_H", "z_d_OH", "z_GCN_O_x_z_GCN_H"]
            + [f"motif_{motif}" for motif in self.motifs]
            + [f"motif_{motif}_x_z_GCN_O" for motif in self.motifs]
        )
        return matrix, names


class TreeFeatureMap:
    def __init__(self) -> None:
        self.motifs: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "TreeFeatureMap":
        self.motifs = sorted(frame["motif"].astype(str).unique())
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        continuous = frame[["GCN_O", "GCN_H", "d_OH_A"]].to_numpy(float)
        one_hot = np.column_stack(
            [(frame["motif"].astype(str).to_numpy() == motif).astype(float) for motif in self.motifs]
        )
        return (
            np.column_stack([continuous, one_hot]),
            ["GCN_O", "GCN_H", "d_OH_A"] + [f"motif_{motif}" for motif in self.motifs],
        )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE_eV": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_eV": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def relative_arrays(frame: pd.DataFrame, prediction_column: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    true_rel = np.empty(len(frame), dtype=float)
    pred_rel = np.empty(len(frame), dtype=float)
    correlations: list[float] = []
    for _, indices in frame.groupby("system", sort=False).groups.items():
        positions = frame.index.get_indexer(indices)
        group = frame.loc[indices]
        true = group["Eads_DFT_eV"].to_numpy(float)
        pred = group[prediction_column].to_numpy(float)
        reference = int(np.argmin(true))
        true_rel[positions] = true - true[reference]
        pred_rel[positions] = pred - pred[reference]
        if len(true) >= 2 and np.ptp(true) > 1e-12 and np.ptp(pred) > 1e-12:
            correlations.append(float(spearmanr(true, pred).statistic))
    return true_rel, pred_rel, correlations


def metric_rows(
    model: str,
    frame: pd.DataFrame,
    prediction_column: str,
    evaluation_set: str,
) -> list[dict[str, object]]:
    result = []
    scopes = [("ALL", frame)]
    scopes.extend((domain, frame[frame["domain"] == domain]) for domain in ("NP", "SLAB"))
    for scope, subset in scopes:
        if subset.empty:
            continue
        absolute = regression_metrics(
            subset["Eads_DFT_eV"].to_numpy(float), subset[prediction_column].to_numpy(float)
        )
        result.append({
            "model": model,
            "evaluation_set": evaluation_set,
            "domain": scope,
            "energy_mode": "absolute",
            "n": len(subset),
            "Spearman_macro": "",
            **absolute,
        })
        true_rel, pred_rel, correlations = relative_arrays(subset.reset_index(drop=True), prediction_column)
        relative = regression_metrics(true_rel, pred_rel)
        result.append({
            "model": model,
            "evaluation_set": evaluation_set,
            "domain": scope,
            "energy_mode": "relative_to_DFT_min_configuration",
            "n": len(subset),
            "Spearman_macro": float(np.mean(correlations)) if correlations else float("nan"),
            **relative,
        })
    return result


def fit_ridge(train: pd.DataFrame, validation: pd.DataFrame):
    feature_map = RidgeFeatureMap().fit(train)
    x_train, names = feature_map.transform(train)
    x_val, _ = feature_map.transform(validation)
    y_train = train["Eads_DFT_eV"].to_numpy(float)
    y_val = validation["Eads_DFT_eV"].to_numpy(float)
    weights = train["sample_weight"].to_numpy(float)
    trials = []
    models = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train, sample_weight=weights)
        prediction = model.predict(x_val)
        score = regression_metrics(y_val, prediction)
        trials.append({"alpha": alpha, **score})
        models.append(model)
    best_index = min(range(len(models)), key=lambda idx: (trials[idx]["MAE_eV"], trials[idx]["RMSE_eV"]))
    return models[best_index], feature_map, names, trials, best_index


def fit_nonlinear(train: pd.DataFrame, validation: pd.DataFrame):
    feature_map = TreeFeatureMap().fit(train)
    x_train, names = feature_map.transform(train)
    x_val, _ = feature_map.transform(validation)
    y_train = train["Eads_DFT_eV"].to_numpy(float)
    y_val = validation["Eads_DFT_eV"].to_numpy(float)
    weights = train["sample_weight"].to_numpy(float)

    try:
        from xgboost import XGBRegressor

        backend = "XGBoost"
        grid = list(product((2, 3, 4), (0.03, 0.05), (1.0, 2.0, 5.0)))
        trials = []
        models = []
        for max_depth, learning_rate, reg_lambda in grid:
            model = XGBRegressor(
                n_estimators=500,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.9,
                colsample_bytree=1.0,
                reg_lambda=reg_lambda,
                objective="reg:squarederror",
                random_state=SEED,
                n_jobs=2,
            )
            model.fit(x_train, y_train, sample_weight=weights, verbose=False)
            score = regression_metrics(y_val, model.predict(x_val))
            trials.append({
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_lambda": reg_lambda,
                **score,
            })
            models.append(model)
    except ImportError:
        backend = "HistGradientBoosting_fallback"
        grid = list(product((7, 15, 31), (0.03, 0.05, 0.1), (1.0, 2.0, 5.0)))
        trials = []
        models = []
        for max_leaf_nodes, learning_rate, l2_regularization in grid:
            model = HistGradientBoostingRegressor(
                max_iter=500,
                max_leaf_nodes=max_leaf_nodes,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
                random_state=SEED,
            )
            model.fit(x_train, y_train, sample_weight=weights)
            score = regression_metrics(y_val, model.predict(x_val))
            trials.append({
                "max_leaf_nodes": max_leaf_nodes,
                "learning_rate": learning_rate,
                "l2_regularization": l2_regularization,
                **score,
            })
            models.append(model)

    best_index = min(range(len(models)), key=lambda idx: (trials[idx]["MAE_eV"], trials[idx]["RMSE_eV"]))
    return backend, models[best_index], feature_map, names, trials, best_index


def export_portable_models(
    output_dir: Path,
    train: pd.DataFrame,
    ridge,
    ridge_map: RidgeFeatureMap,
    ridge_names: list[str],
    nonlinear_name: str,
    nonlinear,
    nonlinear_map: TreeFeatureMap,
    nonlinear_names: list[str],
) -> dict[str, str | None]:
    """Export inference artifacts without pickled custom Python classes.

    The joblib files remain convenient in the training environment, but their
    feature-map objects are defined in this script and can be fragile when the
    bundle is opened from another program or Python environment.  JSON plus
    XGBoost's native model format is safer for cluster-side prediction.
    """
    ranges = {
        name: {
            "min": float(train[name].min()),
            "max": float(train[name].max()),
        }
        for name in ("GCN_O", "GCN_H", "d_OH_A")
    }
    ridge_payload = {
        "format_version": 1,
        "model_type": "Ridge",
        "motifs": ridge_map.motifs,
        "continuous_features": ["GCN_O", "GCN_H", "d_OH_A"],
        "feature_names": ridge_names,
        "scaler_mean": [float(value) for value in ridge_map.scaler.mean_],
        "scaler_scale": [float(value) for value in ridge_map.scaler.scale_],
        "intercept": float(ridge.intercept_),
        "coefficients": [float(value) for value in ridge.coef_],
        "training_descriptor_ranges": ranges,
    }
    ridge_path = output_dir / "ridge_baseline_portable.json"
    ridge_path.write_text(json.dumps(ridge_payload, indent=2) + "\n")

    preprocessing_payload = {
        "format_version": 1,
        "model_type": nonlinear_name,
        "motifs": nonlinear_map.motifs,
        "continuous_features": ["GCN_O", "GCN_H", "d_OH_A"],
        "feature_names": nonlinear_names,
        "training_descriptor_ranges": ranges,
    }
    preprocessing_path = output_dir / "nonlinear_baseline_preprocessing.json"
    preprocessing_path.write_text(json.dumps(preprocessing_payload, indent=2) + "\n")

    native_model_path = None
    if nonlinear_name == "XGBoost":
        native_model_path = output_dir / "xgboost_baseline_native.json"
        nonlinear.save_model(native_model_path)

    return {
        "ridge_portable": ridge_path.name,
        "nonlinear_preprocessing": preprocessing_path.name,
        "nonlinear_native_model": native_model_path.name if native_model_path else None,
    }


def load_m6_validation(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame = frame[frame["split"] == "validation"].copy()
    frame = frame.rename(columns={
        "group": "system",
        "true_energy": "Eads_DFT_eV",
        "pred_energy": "M6_pred_eV",
    })
    return frame


def save_parity(validation: pd.DataFrame, m6: pd.DataFrame | None, output: Path, nonlinear_name: str) -> None:
    model_columns = [("Ridge", "Ridge_pred_eV"), (nonlinear_name, "Nonlinear_pred_eV")]
    if m6 is not None:
        model_columns.insert(0, ("M6-ML", "M6_pred_eV"))
    all_values = [validation["Eads_DFT_eV"].to_numpy(float)]
    all_values.extend(validation[column].to_numpy(float) for _, column in model_columns if column in validation)
    if m6 is not None:
        all_values.extend([m6["Eads_DFT_eV"].to_numpy(float), m6["M6_pred_eV"].to_numpy(float)])
    low = min(float(np.min(values)) for values in all_values) - 0.05
    high = max(float(np.max(values)) for values in all_values) + 0.05

    fig, axes = plt.subplots(len(model_columns), 2, figsize=(9.2, 3.8 * len(model_columns)), squeeze=False)
    for row_idx, (model_name, column) in enumerate(model_columns):
        source = m6 if model_name == "M6-ML" else validation
        for col_idx, domain in enumerate(("NP", "SLAB")):
            ax = axes[row_idx, col_idx]
            subset = source[source["domain"] == domain]
            pred_column = "M6_pred_eV" if model_name == "M6-ML" else column
            ax.scatter(subset["Eads_DFT_eV"], subset[pred_column], s=18, alpha=0.65)
            ax.plot([low, high], [low, high], color="black", lw=1, ls="--")
            score = regression_metrics(
                subset["Eads_DFT_eV"].to_numpy(float), subset[pred_column].to_numpy(float)
            )
            ax.text(
                0.04, 0.96,
                f"n={len(subset)}\nMAE={score['MAE_eV']:.3f} eV\nRMSE={score['RMSE_eV']:.3f} eV\n$R^2$={score['R2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
            )
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_title(f"{model_name}: {'nanoparticles' if domain == 'NP' else 'surfaces'}")
            ax.set_xlabel("DFT adsorption energy (eV)")
            ax.set_ylabel("Predicted adsorption energy (eV)")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--m6-predictions", type=Path, default=DEFAULT_M6_PREDICTIONS,
        help="Saved M6 train/validation predictions used for comparison metrics.",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "reproduced_results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.dataset)
    train = frame[frame["split"] == "train"].copy().reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].copy().reset_index(drop=True)

    ridge, ridge_map, ridge_names, ridge_trials, ridge_best = fit_ridge(train, validation)
    x_val_ridge, _ = ridge_map.transform(validation)
    validation["Ridge_pred_eV"] = ridge.predict(x_val_ridge)

    nonlinear_name, nonlinear, nonlinear_map, nonlinear_names, nonlinear_trials, nonlinear_best = fit_nonlinear(
        train, validation
    )
    x_val_tree, _ = nonlinear_map.transform(validation)
    validation["Nonlinear_pred_eV"] = nonlinear.predict(x_val_tree)

    joblib.dump(
        {"model": ridge, "feature_map": ridge_map, "feature_names": ridge_names},
        args.output_dir / "ridge_baseline.joblib",
    )
    nonlinear_filename = "xgboost_baseline.joblib" if nonlinear_name == "XGBoost" else "hist_gradient_boosting_fallback.joblib"
    joblib.dump(
        {"model": nonlinear, "feature_map": nonlinear_map, "feature_names": nonlinear_names},
        args.output_dir / nonlinear_filename,
    )
    portable_artifacts = export_portable_models(
        args.output_dir,
        train,
        ridge,
        ridge_map,
        ridge_names,
        nonlinear_name,
        nonlinear,
        nonlinear_map,
        nonlinear_names,
    )

    validation.to_csv(args.output_dir / "baseline_validation_predictions.csv", index=False)

    coefficient_rows = [
        {"feature": name, "coefficient": float(value)}
        for name, value in zip(ridge_names, ridge.coef_)
    ]
    coefficient_rows.insert(0, {"feature": "intercept", "coefficient": float(ridge.intercept_)})
    pd.DataFrame(coefficient_rows).to_csv(args.output_dir / "ridge_coefficients.csv", index=False)

    pd.DataFrame(ridge_trials).to_csv(args.output_dir / "ridge_hyperparameter_trials.csv", index=False)
    pd.DataFrame(nonlinear_trials).to_csv(args.output_dir / "nonlinear_hyperparameter_trials.csv", index=False)

    metrics = []
    metrics.extend(metric_rows("Ridge", validation, "Ridge_pred_eV", "M6_validation_all"))
    metrics.extend(metric_rows(nonlinear_name, validation, "Nonlinear_pred_eV", "M6_validation_all"))
    clean = validation[~validation["training_input_overlap"].astype(bool)].copy().reset_index(drop=True)
    metrics.extend(metric_rows("Ridge", clean, "Ridge_pred_eV", "M6_validation_clean_inputs"))
    metrics.extend(metric_rows(nonlinear_name, clean, "Nonlinear_pred_eV", "M6_validation_clean_inputs"))

    m6 = load_m6_validation(args.m6_predictions)
    if m6 is not None:
        metrics.extend(metric_rows("M6-ML", m6, "M6_pred_eV", "M6_validation_all"))
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.output_dir / "baseline_metrics.csv", index=False)

    save_parity(
        validation,
        m6,
        args.output_dir / "baseline_internal_validation_parity.png",
        nonlinear_name,
    )

    settings = {
        "seed": SEED,
        "n_train": len(train),
        "n_validation": len(validation),
        "n_validation_clean_inputs": len(clean),
        "surface_sample_weight": 4.0,
        "ridge_best": ridge_trials[ridge_best],
        "nonlinear_backend": nonlinear_name,
        "nonlinear_best": nonlinear_trials[nonlinear_best],
        "xgboost_available": nonlinear_name == "XGBoost",
        "external_tests_included": False,
        "portable_inference_artifacts": portable_artifacts,
    }
    (args.output_dir / "baseline_settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    summary_lines = [
        "Descriptor baseline comparison against M6-ML",
        "",
        f"Training rows: {len(train)}",
        f"M6 validation rows: {len(validation)}",
        f"Validation rows with an input also present in training: {len(validation) - len(clean)}",
        f"Nonlinear backend: {nonlinear_name}",
        "",
    ]
    if nonlinear_name != "XGBoost":
        summary_lines.extend([
            "WARNING: xgboost is not installed. The nonlinear result is an sklearn",
            "HistGradientBoosting fallback and must not be labelled XGBoost in the paper.",
            "Install xgboost and rerun this script for the requested paper baseline.",
            "",
        ])
    for _, row in metrics_frame[
        (metrics_frame["evaluation_set"] == "M6_validation_all")
        & (metrics_frame["energy_mode"] == "absolute")
    ].iterrows():
        summary_lines.append(
            f"{row['model']:30s} {row['domain']:4s} n={int(row['n']):4d} "
            f"MAE={row['MAE_eV']:.4f} RMSE={row['RMSE_eV']:.4f} R2={row['R2']:.4f}"
        )
    (args.output_dir / "baseline_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
