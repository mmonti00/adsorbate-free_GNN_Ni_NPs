#!/usr/bin/env python3
"""Reproduce the training-environment coverage analysis for the five DFT tests.

The script rebuilds M6 graphs for the released train/validation split and for
the optimized external test structures.  It then evaluates two diagnostics:

1. motif-resolved coverage in (GCN_O, GCN_H);
2. nearest-training distances in the 384-dimensional pooled M6 embedding and
   the 64-dimensional hidden energy-head representation.

Run from any directory with::

    python test_systems/environment_ood_analysis/\
        analyze_test_training_environment_ood.py

PyTorch, PyTorch Geometric, ASE, scikit-learn, pandas, NumPy, and Matplotlib
are required.  By default the four CSV files and the PDF in this directory are
overwritten.  Use ``--output-dir`` to reproduce into a separate directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parent
REPOSITORY_ROOT = TEST_ROOT.parent
TRAINING_DIR = REPOSITORY_ROOT / "training_scripts"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import torch
import torch.nn.functional as F

import analysis_energy_test_systems as external_analysis
import train_M6_energy_model as energy_module


MODEL_PATH = REPOSITORY_ROOT / "models_checkpoints" / "M6_energy_model.pt"
PAPER_SPLIT_MANIFEST = (
    TRAINING_DIR / "paper_splits" / "M6_energy_model_paper_split.json"
)
TRAIN_EXTXYZ = (
    REPOSITORY_ROOT / "models_checkpoints" / "M6_energy_model_all_train_data.extxyz"
)
VALID_EXTXYZ = (
    REPOSITORY_ROOT / "models_checkpoints" / "M6_energy_model_all_valid_data.extxyz"
)

# The reference extxyz files contain the exact paper/checkpoint split,
# including its one documented legacy O*+H*+H* frame.  Activate the narrowly
# scoped compatibility rule in the training module before rebuilding graphs.
if not PAPER_SPLIT_MANIFEST.is_file():
    raise FileNotFoundError(PAPER_SPLIT_MANIFEST)
energy_module.FIXED_SPLIT_JSON = str(PAPER_SPLIT_MANIFEST)

SYSTEMS = (
    ("Ni55", "NP"),
    ("Ni91", "NP"),
    ("Ni130", "NP"),
    ("Ni111_4x3", "SLAB"),
    ("Ni100_4x4", "SLAB"),
)
DISPLAY_NAMES = {
    "Ni55": r"Ni$_{55}$",
    "Ni91": r"Ni$_{91}$",
    "Ni130": r"Ni$_{130}$",
    "Ni111_4x3": r"Ni(111)-4$\times$3",
    "Ni100_4x4": r"Ni(100)-4$\times$4",
}
TEST_BARE_FILES = {
    "Ni55": "Ni55_MACE_opt.xyz",
    "Ni91": "Ni91_MACE_opt.xyz",
    "Ni130": "Ni130_MACE_opt.xyz",
    "Ni111_4x3": "Ni111_4x3_MACE_opt.extxyz",
    "Ni100_4x4": "Ni100_4x4_MACE_opt.extxyz",
}

GCN_CN_MAX = {"T": 12.0, "B": 18.0, "H": 22.0, "F": 26.0}
NI_NI_CUTOFF = 3.0
GCN_EXACT_TOL = 1.0e-8
GCN_NEAR_TOL = 0.15
MIN_MOTIF_CALIBRATION = 10

REPRESENTATIONS = ("pooled_384", "head_hidden_64")
STATUS_ORDER = (
    "exact_environment",
    "near_environment",
    "within_marginal_ranges",
    "outside_training_range",
    "motif_unseen",
)


def torch_load(path: Path, device: str):
    """Load old and new PyTorch checkpoints without version-specific failure."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(device: str):
    checkpoint = torch_load(MODEL_PATH, device)
    model = energy_module.BareSiteGCNN(
        node_dim=int(checkpoint["node_dim"]),
        edge_dim=int(checkpoint["edge_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_training_bare_references() -> dict[str, object]:
    references = {}
    for label, relative_path in energy_module.INITIAL_NI_FILES.items():
        path = (TRAINING_DIR / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        references[label] = read(path)
    return references


def load_test_bare_reference(system: str):
    path = (
        REPOSITORY_ROOT
        / "ML_relaxed_bare_geom"
        / "MACE"
        / "testing"
        / TEST_BARE_FILES[system]
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return read(path)


def graph_from_anchors(bare, o_anchors: list[int], h_anchors: list[int], domain: str):
    """Build the exact bare-Ni graph used by M6 for supplied anchor sets."""
    positions = bare.get_positions()
    ni_indices = list(range(len(bare)))
    all_anchors = sorted(set(o_anchors) | set(h_anchors))
    local_cutoff = energy_module.get_local_cutoff(domain)
    edge_cutoff = energy_module.get_edge_cutoff(domain)

    if domain == "SLAB":
        _images, x, local_positions, _keep = energy_module.build_slab_local_graph(
            ni_indices,
            positions,
            o_anchors,
            h_anchors,
            positions,
            bare.cell,
            bare.pbc,
            local_cutoff,
            edge_cutoff,
        )
        edge_index, edge_attr = energy_module.build_edges_and_edge_attr(
            local_positions,
            bare.cell,
            [False, False, False],
            edge_cutoff,
        )
    else:
        keep = energy_module.select_bare_local_environment(
            ni_indices,
            positions,
            all_anchors,
            bare.cell,
            bare.pbc,
            local_cutoff,
        )
        local_positions = energy_module.unwrap_positions_around_reference(
            positions,
            keep,
            bare.cell,
            bare.pbc,
            ref_idx=all_anchors[0],
        )
        x = energy_module.build_node_features(
            keep,
            positions,
            o_anchors,
            h_anchors,
            bare.cell,
            bare.pbc,
            edge_cutoff=edge_cutoff,
        )
        edge_index, edge_attr = energy_module.build_edges_and_edge_attr(
            local_positions,
            bare.cell,
            bare.pbc,
            edge_cutoff,
        )

    return energy_module.Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=torch.tensor(local_positions, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def model_representations(model, graph, device: str):
    """Return prediction, pooled representation, and hidden energy-head vector."""
    graph = graph.to(device)
    batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        x_in = graph.x
        h = F.relu(model.conv1(graph.x, graph.edge_index, graph.edge_attr))
        h = F.relu(model.conv2(h, graph.edge_index, graph.edge_attr))
        h = F.relu(model.conv3(h, graph.edge_index, graph.edge_attr))

        g_global = energy_module.global_mean_pool(h, batch)
        weights_o = (
            torch.exp(-x_in[:, 4:5] / energy_module.POOL_SIGMA)
            + energy_module.ANCHOR_BOOST * x_in[:, 1:2]
        )
        weights_h = (
            torch.exp(-x_in[:, 5:6] / energy_module.POOL_SIGMA)
            + energy_module.ANCHOR_BOOST * x_in[:, 2:3]
        )
        g_o = model.weighted_mean_pool(h, weights_o, batch)
        g_h = model.weighted_mean_pool(h, weights_h, batch)
        pooled = torch.cat([g_global, g_o, g_h], dim=-1)
        hidden = F.relu(model.energy_head[0](pooled))
        prediction = model.energy_head[2](hidden).squeeze(-1)

    return (
        float(prediction.cpu().item()),
        pooled.cpu().numpy().reshape(-1),
        hidden.cpu().numpy().reshape(-1),
    )


def periodic_coordination(bare, cell, pbc):
    work = bare.copy()
    if np.any(pbc):
        work.set_cell(cell)
    work.set_pbc(pbc)
    i_idx, j_idx, shifts = neighbor_list(
        "ijS", work, NI_NI_CUTOFF, self_interaction=False
    )
    neighbors: list[list[tuple[int, tuple[int, int, int]]]] = [
        [] for _ in range(len(work))
    ]
    for i, j, shift in zip(i_idx, j_idx, shifts):
        neighbors[int(i)].append((int(j), tuple(int(value) for value in shift)))
    coordination = np.asarray([len(items) for items in neighbors], dtype=float)
    return work, neighbors, coordination


def canonical_anchor_images(atoms, anchors):
    """Place all anchors in one local periodic image around the first anchor."""
    anchors = sorted(int(anchor) for anchor in anchors)
    if not np.any(atoms.pbc):
        return [(anchor, (0, 0, 0)) for anchor in anchors]
    reference = anchors[0]
    inverse_cell = np.linalg.inv(atoms.cell.array)
    result = []
    for anchor in anchors:
        raw_vector = atoms.positions[anchor] - atoms.positions[reference]
        mic_vector, _ = find_mic(raw_vector, atoms.cell, atoms.pbc)
        shift = np.rint((mic_vector - raw_vector) @ inverse_cell).astype(int)
        result.append((anchor, tuple(int(value) for value in shift)))
    return result


def site_gcn(atoms, anchors, site_type: str, neighbors, coordination) -> float:
    """Eq. S18, retaining distinct periodic neighbor images."""
    anchor_images = canonical_anchor_images(atoms, anchors)
    neighbor_images: set[tuple[int, tuple[int, int, int]]] = set()
    for anchor, anchor_shift in anchor_images:
        anchor_shift_array = np.asarray(anchor_shift, dtype=int)
        for neighbor, neighbor_shift in neighbors[anchor]:
            combined_shift = anchor_shift_array + np.asarray(neighbor_shift, dtype=int)
            neighbor_images.add(
                (neighbor, tuple(int(value) for value in combined_shift))
            )
    neighbor_images.difference_update(anchor_images)
    return float(
        sum(float(coordination[index]) for index, _shift in neighbor_images)
        / GCN_CN_MAX[site_type]
    )


def add_descriptors(record: dict, bare, o_anchors, h_anchors, motif, cell, pbc):
    work, neighbors, coordination = periodic_coordination(bare, cell, pbc)
    record["gcn_o"] = site_gcn(
        work, o_anchors, motif[0], neighbors, coordination
    )
    record["gcn_h"] = site_gcn(
        work, h_anchors, motif[1], neighbors, coordination
    )


def config_sort_key(path: Path):
    match = re.search(r"conf(\d+)", path.name)
    return int(match.group(1)) if match else path.name


def build_split_records(
    split_name: str,
    path: Path,
    references: dict[str, object],
    model,
    device: str,
) -> list[dict]:
    frames = read(path, index=":")
    groups = energy_module.build_group_labels(frames, list(references))
    records = []
    for index, (atoms, group) in enumerate(zip(frames, groups)):
        domain = "SLAB" if np.any(atoms.pbc) else "NP"
        graph, meta = energy_module.process_frame(
            atoms,
            references,
            group,
            index,
            domain,
            4.0 if domain == "SLAB" else 1.0,
        )
        prediction, pooled, hidden = model_representations(model, graph, device)
        record = {
            "set": split_name,
            "domain": domain,
            "system": group,
            "config": str(atoms.info.get("config_label", f"{split_name}_{index}")),
            "motif": meta["motif"],
            "h_short": float(energy_module.NI_H_CUTOFF),
            "model_prediction_eV": prediction,
            "pooled_384": pooled,
            "head_hidden_64": hidden,
        }
        add_descriptors(
            record,
            references[group],
            meta["o_anchor_idx"],
            meta["h_anchor_idx"],
            meta["motif"],
            atoms.cell,
            atoms.pbc,
        )
        records.append(record)
    return records


def build_test_records(model, device: str) -> list[dict]:
    records = []
    for system, domain in SYSTEMS:
        bare = load_test_bare_reference(system)
        files = sorted((TEST_ROOT / system / "opt_xyz").glob("conf*_opt.xyz"), key=config_sort_key)
        if not files:
            raise FileNotFoundError(f"No optimized structures found for {system}")
        for path in files:
            atoms = read(path)
            cell = atoms.cell.array if atoms.cell.volume > 1.0e-12 else None
            pbc = np.asarray(atoms.pbc, dtype=bool)
            motif, o_set, h_set = external_analysis.get_final_site(
                atoms.get_chemical_symbols(),
                atoms.get_positions(),
                cell,
                pbc,
            )
            o_anchors = sorted(int(value) for value in o_set)
            h_anchors = sorted(int(value) for value in h_set)
            graph = graph_from_anchors(bare, o_anchors, h_anchors, domain)
            prediction, pooled, hidden = model_representations(model, graph, device)
            config_match = re.search(r"(conf\d+)", path.name)
            record = {
                "set": "test",
                "domain": domain,
                "system": system,
                "config": config_match.group(1) if config_match else path.stem,
                "motif": motif,
                "h_short": float(external_analysis.NI_H_CUTOFF),
                "model_prediction_eV": prediction,
                "pooled_384": pooled,
                "head_hidden_64": hidden,
            }
            add_descriptors(
                record,
                bare,
                o_anchors,
                h_anchors,
                motif,
                bare.cell,
                bare.pbc,
            )
            records.append(record)
    return records


def gcn_coverage(train: list[dict], tests: list[dict]) -> None:
    for test in tests:
        same = [
            row
            for row in train
            if row["domain"] == test["domain"] and row["motif"] == test["motif"]
        ]
        test["training_same_motif_count"] = len(same)
        if not same:
            test.update(
                gcn_status="motif_unseen",
                nearest_train_gcn_config=np.nan,
                delta_gcn_o=np.nan,
                delta_gcn_h=np.nan,
                nearest_max_delta=np.nan,
                inside_training_gcn_ranges=False,
            )
            continue

        train_values = np.asarray([[row["gcn_o"], row["gcn_h"]] for row in same])
        query = np.asarray([test["gcn_o"], test["gcn_h"]])
        absolute_delta = np.abs(train_values - query[None, :])
        maximum_delta = absolute_delta.max(axis=1)
        nearest_index = int(np.argmin(maximum_delta))
        nearest_delta = absolute_delta[nearest_index]
        nearest_maximum = float(maximum_delta[nearest_index])

        inside = bool(
            train_values[:, 0].min() <= query[0] <= train_values[:, 0].max()
            and train_values[:, 1].min() <= query[1] <= train_values[:, 1].max()
        )
        if nearest_maximum <= GCN_EXACT_TOL:
            status = "exact_environment"
        elif nearest_maximum <= GCN_NEAR_TOL:
            status = "near_environment"
        elif inside:
            status = "within_marginal_ranges"
        else:
            status = "outside_training_range"

        test.update(
            gcn_status=status,
            nearest_train_gcn_config=same[nearest_index]["config"],
            delta_gcn_o=float(nearest_delta[0]),
            delta_gcn_h=float(nearest_delta[1]),
            nearest_max_delta=nearest_maximum,
            inside_training_gcn_ranges=inside,
        )


def nearest(query: np.ndarray, reference: np.ndarray, metric: str):
    distances = pairwise_distances(query.reshape(1, -1), reference, metric=metric)[0]
    index = int(np.argmin(distances))
    return float(distances[index]), index


def transformed_representations(train, validation, tests):
    """Fit training-only standardization/PCA and return transformed arrays."""
    transformed = {}
    for representation in REPRESENTATIONS:
        train_values = np.stack([row[representation] for row in train])
        validation_values = np.stack([row[representation] for row in validation])
        test_values = np.stack([row[representation] for row in tests])
        scaler = StandardScaler().fit(train_values)
        pca = PCA(n_components=0.95, whiten=True, svd_solver="full")
        train_pca = pca.fit_transform(scaler.transform(train_values))
        validation_pca = pca.transform(scaler.transform(validation_values))
        test_pca = pca.transform(scaler.transform(test_values))
        transformed[representation] = {
            "cosine": (train_values, validation_values, test_values),
            f"pca95_whitened_{pca.n_components_}d": (
                train_pca,
                validation_pca,
                test_pca,
            ),
        }
    return transformed


def calibration_distances(
    train,
    validation,
    train_values,
    validation_values,
    metric: str,
    domain: str,
    motif: str | None,
) -> np.ndarray:
    train_indices = [
        i
        for i, row in enumerate(train)
        if row["domain"] == domain and (motif is None or row["motif"] == motif)
    ]
    validation_indices = [
        i
        for i, row in enumerate(validation)
        if row["domain"] == domain and (motif is None or row["motif"] == motif)
    ]
    if not train_indices or not validation_indices:
        return np.asarray([], dtype=float)
    reference = train_values[train_indices]
    return np.asarray(
        [nearest(validation_values[i], reference, metric)[0] for i in validation_indices],
        dtype=float,
    )


def embedding_diagnostics(train, validation, tests) -> pd.DataFrame:
    transformed = transformed_representations(train, validation, tests)
    rows = []
    for representation in REPRESENTATIONS:
        for metric_name, arrays in transformed[representation].items():
            train_values, validation_values, test_values = arrays
            distance_metric = "cosine" if metric_name == "cosine" else "euclidean"

            domain_calibration = {}
            motif_calibration = {}
            for domain in sorted({row["domain"] for row in train}):
                domain_distances = calibration_distances(
                    train,
                    validation,
                    train_values,
                    validation_values,
                    distance_metric,
                    domain,
                    None,
                )
                domain_calibration[domain] = domain_distances
                motifs = sorted(
                    {row["motif"] for row in validation if row["domain"] == domain}
                )
                for motif in motifs:
                    motif_calibration[(domain, motif)] = calibration_distances(
                        train,
                        validation,
                        train_values,
                        validation_values,
                        distance_metric,
                        domain,
                        motif,
                    )

            for test_index, test in enumerate(tests):
                key = (test["domain"], test["motif"])
                motif_distances = motif_calibration.get(key, np.asarray([]))
                if len(motif_distances) >= MIN_MOTIF_CALIBRATION:
                    calibration = motif_distances
                    scope = "domain+motif"
                else:
                    calibration = domain_calibration[test["domain"]]
                    scope = "domain_fallback"
                threshold = float(np.quantile(calibration, 0.95))

                train_indices = [
                    i
                    for i, row in enumerate(train)
                    if row["domain"] == test["domain"]
                    and row["motif"] == test["motif"]
                ]
                if train_indices:
                    distance, local_index = nearest(
                        test_values[test_index],
                        train_values[train_indices],
                        distance_metric,
                    )
                    nearest_label = train[train_indices[local_index]]["config"]
                    is_ood = bool(distance > threshold)
                else:
                    distance = np.nan
                    nearest_label = np.nan
                    is_ood = True

                rows.append(
                    {
                        "set": "test",
                        "domain": test["domain"],
                        "system": test["system"],
                        "config": test["config"],
                        "motif": test["motif"],
                        "h_short": test["h_short"],
                        "gcn_o": test["gcn_o"],
                        "gcn_h": test["gcn_h"],
                        "model_prediction_eV": test["model_prediction_eV"],
                        "representation": representation,
                        "metric": metric_name,
                        "distance": distance,
                        "q95_validation": threshold,
                        "n_calibration": int(len(calibration)),
                        "calibration_scope": scope,
                        "ood_q95": is_ood,
                        "nearest_train_config": nearest_label,
                    }
                )
    return pd.DataFrame(rows)


def attach_consensus(tests: list[dict], embedding_details: pd.DataFrame) -> None:
    for test in tests:
        subset = embedding_details[
            (embedding_details["system"] == test["system"])
            & (embedding_details["config"] == test["config"])
        ]
        head = subset[subset["representation"] == "head_hidden_64"]
        test["energy_head_consensus_ood"] = bool(head["ood_q95"].all())
        test["all_embedding_consensus_ood"] = bool(subset["ood_q95"].all())


def test_detail_frame(tests: list[dict]) -> pd.DataFrame:
    columns = [
        "set",
        "domain",
        "system",
        "config",
        "motif",
        "h_short",
        "gcn_o",
        "gcn_h",
        "model_prediction_eV",
        "training_same_motif_count",
        "gcn_status",
        "nearest_train_gcn_config",
        "delta_gcn_o",
        "delta_gcn_h",
        "nearest_max_delta",
        "inside_training_gcn_ranges",
        "energy_head_consensus_ood",
        "all_embedding_consensus_ood",
    ]
    return pd.DataFrame([{column: row[column] for column in columns} for row in tests])


def test_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, _domain in SYSTEMS:
        group = details[details["system"] == system]
        counts = Counter(group["gcn_status"])
        n = len(group)
        clear = counts["outside_training_range"] + counts["motif_unseen"]
        row = {"system": system, "n": n}
        mappings = (
            ("gcn_exact_environment", counts["exact_environment"]),
            ("gcn_near_environment", counts["near_environment"]),
            ("gcn_within_range_gap", counts["within_marginal_ranges"]),
            ("gcn_extrapolation", counts["outside_training_range"]),
            ("gcn_motif_unseen", counts["motif_unseen"]),
        )
        for prefix, value in mappings:
            row[f"{prefix}_n"] = int(value)
            row[f"{prefix}_pct"] = 100.0 * value / n
        row["inside_training_gcn_ranges_pct"] = float(
            100.0 * group["inside_training_gcn_ranges"].mean()
        )
        row["clear_gcn_ood_n"] = int(clear)
        row["clear_gcn_ood_pct"] = 100.0 * clear / n
        energy_count = int(group["energy_head_consensus_ood"].sum())
        all_count = int(group["all_embedding_consensus_ood"].sum())
        row["energy_head_embedding_ood_n"] = energy_count
        row["energy_head_embedding_ood_pct"] = 100.0 * energy_count / n
        row["all_embedding_consensus_ood_n"] = all_count
        row["all_embedding_consensus_ood_pct"] = 100.0 * all_count / n
        rows.append(row)
    return pd.DataFrame(rows)


def embedding_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (representation, metric, system), group in details.groupby(
        ["representation", "metric", "system"], sort=False
    ):
        n = len(group)
        n_ood = int(group["ood_q95"].sum())
        rows.append(
            {
                "representation": representation,
                "metric": metric,
                "system": system,
                "n": n,
                "n_ood_q95": n_ood,
                "ood_q95_pct": 100.0 * n_ood / n,
                "median_distance": float(group["distance"].median()),
                "q95_validation": float(group["q95_validation"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def make_figure(details: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
        }
    )
    systems = [system for system, _domain in SYSTEMS]
    x = np.arange(len(systems), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13.7, 4.9), sharey=True)

    status_style = (
        ("exact_environment", "Exact GCN pair", "#1b9e77"),
        ("near_environment", r"Near (max$|\Delta$GCN$|\leq 0.15$)", "#4c78a8"),
        ("within_marginal_ranges", "Inside marginal ranges", "#f2a43a"),
        ("outside_training_range", "Outside training range", "#d95f5f"),
        ("motif_unseen", "Motif unseen", "#555555"),
    )
    bottom = np.zeros(len(systems))
    for status, label, color in status_style:
        values = np.asarray(
            [100.0 * (details[(details.system == s) & (details.gcn_status == status)].shape[0]) / (details.system == s).sum() for s in systems]
        )
        axes[0].bar(x, values, width=0.72, bottom=bottom, color=color, label=label)
        bottom += values
    axes[0].legend(loc="upper left", frameon=True, edgecolor="0.7")
    axes[0].set_ylabel("Configurations (%)", fontsize=15)

    right_specs = (
        ("clear_gcn_ood", "GCN extrapolation", "#d95f5f"),
        ("energy_head_embedding_ood", "Energy-head embedding", "#4c78a8"),
        ("all_embedding_consensus_ood", "All-embedding consensus", "#7b5195"),
    )
    width = 0.24
    for offset_index, (prefix, label, color) in enumerate(right_specs):
        values = summary[f"{prefix}_pct"].to_numpy(float)
        counts = summary[f"{prefix}_n"].to_numpy(int)
        positions = x + (offset_index - 1) * width
        axes[1].bar(positions, values, width=width, color=color, label=label)
        for position, value, count in zip(positions, values, counts):
            if count:
                axes[1].text(position, value + 1.5, str(count), ha="center", va="bottom", fontsize=9)
    axes[1].legend(loc="upper left", frameon=True, edgecolor="0.7")

    for axis in axes:
        axis.set_xticks(x, [DISPLAY_NAMES[system] for system in systems], rotation=20, ha="right")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", color="0.88", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=3.2)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def validate_inputs() -> None:
    required = [MODEL_PATH, TRAIN_EXTXYZ, VALID_EXTXYZ]
    for system, _domain in SYSTEMS:
        required.append(TEST_ROOT / system / "opt_xyz")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_inputs()

    print(f"Loading M6 model on {args.device} ...")
    model = load_model(args.device)
    references = load_training_bare_references()
    print("Building released training graphs ...")
    train = build_split_records("train", TRAIN_EXTXYZ, references, model, args.device)
    print("Building released validation graphs ...")
    validation = build_split_records(
        "validation", VALID_EXTXYZ, references, model, args.device
    )
    print("Building external-test graphs ...")
    tests = build_test_records(model, args.device)
    print(f"Records: train={len(train)}, validation={len(validation)}, test={len(tests)}")

    gcn_coverage(train, tests)
    learned = embedding_diagnostics(train, validation, tests)
    attach_consensus(tests, learned)
    details = test_detail_frame(tests)
    summary = test_summary(details)
    learned_summary = embedding_summary(learned)

    details.to_csv(output_dir / "test_environment_ood_details.csv", index=False)
    summary.to_csv(output_dir / "test_environment_ood_summary.csv", index=False)
    learned.to_csv(output_dir / "learned_embedding_ood_details.csv", index=False)
    learned_summary.to_csv(
        output_dir / "learned_embedding_metric_summary.csv", index=False
    )
    make_figure(details, summary, output_dir / "test_training_environment_ood.pdf")

    print(summary.to_string(index=False))
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
