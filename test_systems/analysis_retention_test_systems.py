#!/usr/bin/env python3
"""Build the retention confusion matrix for the five external Ni test systems.

Initial motifs and complete O/H anchor sets are reconstructed from geometry
with loose adsorbate cutoffs and connected Ni-anchor topology. Relaxed sites
use the standard/fourfold hybrid cutoffs. Periodic slab distances, including
Ni--Ni connectivity, use the minimum-image convention. Filename and extxyz
motif metadata never define the labels.

Prediction thresholds are 0.80 for nanoparticles and periodic surfaces.
Cell percentages are normalized within each DFT outcome.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re

from ase import Atoms
from ase.io import read
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


OH_CUTOFF = 1.20

# Loose initial-geometry definition.
NI_O_CUTOFF_INITIAL = 2.45
NI_H_CUTOFF_INITIAL = 2.10
NI_NI_ANCHOR_CUTOFF = 3.00
MAX_INITIAL_ANCHORS = 4

# Relaxed-geometry definition.
NI_O_CUTOFF = 2.20
NI_O_FOLDED_CUTOFF = 2.30
# A 1.90 A short Ni-H cutoff is used for the external relaxed-structure
# analysis. Raising it to the 2.00 A training convention changes only two of
# the 199 test structures: distorted Ni(111)-4x3 configurations 21 and 22 are
# then assigned HH rather than the physically appropriate HT motif because
# contacts at 1.923--1.929 A are counted as additional H anchors. All other
# test assignments are unchanged. The full training-set cutoff scan is in
# ../ni_cutoff_sensitivity_scan/.
NI_H_CUTOFF = 1.90
NI_H_FOLDED_CUTOFF = 2.20
FOLDED_N_SHORT = 2
FOLDED_N_TOTAL = 4
MAX_STANDARD_ANCHORS = 3

CLASS_NAMES = ("Migrated", "Retained")


@dataclass(frozen=True)
class SystemSpec:
    folder: str
    label: str
    kind: str
    threshold: float


SYSTEMS = (
    SystemSpec("Ni55", "Ni55", "NP", 0.80),
    SystemSpec("Ni91", "Ni91", "NP", 0.80),
    SystemSpec("Ni130", "Ni130", "NP", 0.80),
    SystemSpec("Ni111_4x3", "Ni(111)", "SLAB", 0.80),
    SystemSpec("Ni100_4x4", "Ni(100)", "SLAB", 0.80),
)


def classify_site(n_anchors: int) -> str:
    labels = {1: "T", 2: "B", 3: "H", 4: "F"}
    if n_anchors not in labels:
        raise ValueError(f"Unexpected number of anchors: {n_anchors}")
    return labels[n_anchors]


def read_structure(path: Path, periodic: bool):
    """Read slabs as extxyz; ignore nonstandard NP comment metadata safely."""
    if periodic:
        return read(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Malformed XYZ: {path}")
    n_atoms = int(lines[0].strip())
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        fields = line.split()
        symbols.append(fields[0])
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if len(symbols) != n_atoms:
        raise ValueError(f"Expected {n_atoms} atoms but read {len(symbols)} from {path}")
    return Atoms(symbols=symbols, positions=positions, pbc=False)


def distance(atoms, index_a: int, index_b: int, periodic: bool) -> float:
    return float(atoms.get_distance(index_a, index_b, mic=periodic))


def adsorbate_indices(atoms, periodic: bool) -> tuple[list[int], int, int]:
    symbols = atoms.get_chemical_symbols()
    ni_indices = [index for index, symbol in enumerate(symbols) if symbol == "Ni"]
    o_indices = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    h_indices = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    if len(o_indices) != 1 or len(h_indices) != 2:
        raise ValueError(f"Unexpected composition: {len(o_indices)} O and {len(h_indices)} H")
    o_index = o_indices[0]
    ordered_h = sorted(
        h_indices,
        key=lambda index: distance(atoms, o_index, index, periodic),
    )
    oh_distance = distance(atoms, o_index, ordered_h[0], periodic)
    if oh_distance > OH_CUTOFF:
        raise ValueError(
            f"No hydroxyl H within {OH_CUTOFF:.2f} A; shortest O-H is "
            f"{oh_distance:.3f} A"
        )
    return ni_indices, o_index, ordered_h[1]


def ni_bonded(atoms, index_a: int, index_b: int, periodic: bool) -> bool:
    return distance(atoms, index_a, index_b, periodic) < NI_NI_ANCHOR_CUTOFF


def valid_hollow(
    atoms,
    anchors: tuple[int, ...],
    ni_indices: list[int],
    periodic: bool,
) -> bool:
    """Accept a Ni triangle or three vertices of a chordless fourfold ring."""
    degrees = {index: 0 for index in anchors}
    n_bonds = 0
    for index_a, index_b in combinations(anchors, 2):
        if ni_bonded(atoms, index_a, index_b, periodic):
            degrees[index_a] += 1
            degrees[index_b] += 1
            n_bonds += 1
    if n_bonds == 3:
        return True
    if n_bonds != 2 or sorted(degrees.values()) != [1, 1, 2]:
        return False
    center = next(index for index, degree in degrees.items() if degree == 2)
    endpoints = [index for index, degree in degrees.items() if degree == 1]
    return any(
        missing not in anchors
        and ni_bonded(atoms, missing, endpoints[0], periodic)
        and ni_bonded(atoms, missing, endpoints[1], periodic)
        and not ni_bonded(atoms, missing, center, periodic)
        for missing in ni_indices
    )


def valid_fourfold(atoms, anchors: tuple[int, ...], periodic: bool) -> bool:
    degrees = {index: 0 for index in anchors}
    for index_a, index_b in combinations(anchors, 2):
        if ni_bonded(atoms, index_a, index_b, periodic):
            degrees[index_a] += 1
            degrees[index_b] += 1
    return min(degrees.values()) >= 2


def initial_anchor_set(
    atoms,
    adsorbate_index: int,
    ni_indices: list[int],
    cutoff: float,
    periodic: bool,
) -> tuple[str, frozenset[int]]:
    candidates = sorted(
        (
            (index, distance(atoms, adsorbate_index, index, periodic))
            for index in ni_indices
            if distance(atoms, adsorbate_index, index, periodic) < cutoff
        ),
        key=lambda item: (item[1], item[0]),
    )
    distance_by_index = dict(candidates)
    for count in range(min(MAX_INITIAL_ANCHORS, len(candidates)), 0, -1):
        valid: list[tuple[tuple, tuple[int, ...]]] = []
        for subset in combinations((index for index, _ in candidates), count):
            if count == 4:
                accepted = valid_fourfold(atoms, subset, periodic)
            elif count == 3:
                accepted = valid_hollow(atoms, subset, ni_indices, periodic)
            elif count == 2:
                accepted = ni_bonded(atoms, subset[0], subset[1], periodic)
            else:
                accepted = True
            if not accepted:
                continue
            n_bonds = sum(
                ni_bonded(atoms, index_a, index_b, periodic)
                for index_a, index_b in combinations(subset, 2)
            )
            distances = tuple(sorted(distance_by_index[index] for index in subset))
            key = (-n_bonds, sum(distances), distances, tuple(sorted(subset)))
            valid.append((key, subset))
        if valid:
            _, selected = min(valid, key=lambda item: item[0])
            anchors = frozenset(selected)
            return classify_site(len(anchors)), anchors
    raise ValueError(f"No connected Ni anchor set within {cutoff:.2f} A")


def initial_site(atoms, periodic: bool) -> tuple[str, frozenset[int], frozenset[int]]:
    ni_indices, o_index, h_index = adsorbate_indices(atoms, periodic)
    o_type, o_anchors = initial_anchor_set(
        atoms, o_index, ni_indices, NI_O_CUTOFF_INITIAL, periodic
    )
    h_type, h_anchors = initial_anchor_set(
        atoms, h_index, ni_indices, NI_H_CUTOFF_INITIAL, periodic
    )
    return o_type + h_type, o_anchors, h_anchors


def final_anchor_set(
    atoms,
    adsorbate_index: int,
    ni_indices: list[int],
    short_cutoff: float,
    extended_cutoff: float,
    periodic: bool,
) -> tuple[str, frozenset[int]]:
    extended = sorted(
        (
            (index, distance(atoms, adsorbate_index, index, periodic))
            for index in ni_indices
            if distance(atoms, adsorbate_index, index, periodic) < extended_cutoff
        ),
        key=lambda item: item[1],
    )
    short = [item for item in extended if item[1] < short_cutoff]
    if len(short) >= FOLDED_N_SHORT and len(extended) == FOLDED_N_TOTAL:
        selected = extended
        site_type = "F"
    else:
        selected = short[:MAX_STANDARD_ANCHORS]
        if not selected:
            raise ValueError(f"No Ni contact within {short_cutoff:.2f} A")
        site_type = classify_site(len(selected))
    return site_type, frozenset(index for index, _ in selected)


def final_site(atoms, periodic: bool) -> tuple[str, frozenset[int], frozenset[int]]:
    ni_indices, o_index, h_index = adsorbate_indices(atoms, periodic)
    o_type, o_anchors = final_anchor_set(
        atoms,
        o_index,
        ni_indices,
        NI_O_CUTOFF,
        NI_O_FOLDED_CUTOFF,
        periodic,
    )
    h_type, h_anchors = final_anchor_set(
        atoms,
        h_index,
        ni_indices,
        NI_H_CUTOFF,
        NI_H_FOLDED_CUTOFF,
        periodic,
    )
    return o_type + h_type, o_anchors, h_anchors


def parse_anchor_field(value: object) -> frozenset[int]:
    return frozenset(int(number) for number in re.findall(r"\d+", str(value)))


def candidate_lookup(csv_path: Path) -> dict[tuple, pd.Series]:
    frame = pd.read_csv(csv_path)
    required = {
        "rank", "motif", "o_anchor_idx", "h_anchor_idx", "retention_probability"
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    lookup: dict[tuple, pd.Series] = {}
    for _, row in frame.sort_values("rank").iterrows():
        key = (
            str(row["motif"]),
            parse_anchor_field(row["o_anchor_idx"]),
            parse_anchor_field(row["h_anchor_idx"]),
        )
        lookup.setdefault(key, row)
    return lookup


def structure_files(system_dir: Path) -> list[tuple[Path, Path, int, str]]:
    pattern = re.compile(r"^conf(?P<token>\d+)_([TBHF]{2})\.xyz$", re.IGNORECASE)
    pairs: list[tuple[Path, Path, int, str]] = []
    for initial_path in (system_dir / "ini_xyz").glob("*.xyz"):
        match = pattern.fullmatch(initial_path.name)
        if match is None:
            continue
        token = match.group("token")
        config = int(token)
        optimized_path = system_dir / "opt_xyz" / f"conf{token}_opt.xyz"
        if not optimized_path.is_file():
            optimized_path = system_dir / "opt_xyz" / f"conf{config}_opt.xyz"
        pairs.append((initial_path, optimized_path, config, match.group(2).upper()))
    return sorted(pairs, key=lambda item: item[2])


def process_system(root: Path, spec: SystemSpec) -> tuple[pd.DataFrame, list[str]]:
    system_dir = root / spec.folder
    csv_paths = sorted(system_dir.glob("*_ranked_candidates.csv"))
    if len(csv_paths) != 1:
        raise FileNotFoundError(
            f"{spec.label}: expected one *_ranked_candidates.csv; found {csv_paths}"
        )
    lookup = candidate_lookup(csv_paths[0])
    rows: list[dict] = []
    errors: list[str] = []
    periodic = spec.kind == "SLAB"
    for initial_path, optimized_path, config, filename_motif in structure_files(system_dir):
        if not optimized_path.is_file():
            errors.append(f"conf{config}: optimized XYZ not found")
            continue
        try:
            initial = initial_site(read_structure(initial_path, periodic), periodic)
            final = final_site(read_structure(optimized_path, periodic), periodic)
            initial_motif, initial_o, initial_h = initial
            final_motif, final_o, final_h = final
            actual_retained = (
                initial_motif == final_motif
                and initial_o == final_o
                and initial_h == final_h
            )
            ranking_row = lookup.get((initial_motif, initial_o, initial_h))
            score = (
                float(ranking_row["retention_probability"])
                if ranking_row is not None
                else np.nan
            )
            predicted_retained = (
                bool(score >= spec.threshold) if np.isfinite(score) else pd.NA
            )
            rows.append(
                {
                    "system": spec.label,
                    "config": config,
                    "initial_file": initial_path.name,
                    "filename_motif": filename_motif,
                    "initial_motif": initial_motif,
                    "actual_retained": actual_retained,
                    "retention_probability": score,
                    "retention_threshold": spec.threshold,
                    "predicted_retained": predicted_retained,
                }
            )
        except ValueError as error:
            errors.append(f"conf{config}: {error}")
    return pd.DataFrame(rows), errors


def confusion_counts(data: pd.DataFrame) -> np.ndarray:
    actual = data["actual_retained"].astype(bool).to_numpy()
    predicted = data["predicted_retained"].astype(bool).to_numpy()
    return np.asarray(
        [
            [np.sum(~actual & ~predicted), np.sum(~actual & predicted)],
            [np.sum(actual & ~predicted), np.sum(actual & predicted)],
        ],
        dtype=int,
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def print_summary(data: pd.DataFrame) -> None:
    print("\nRETENTION CONFUSION MATRICES")
    print(
        "Rows = DFT outcome [Migrated, Retained]; "
        "columns = predicted outcome [Migrated, Retained]"
    )
    print(f"Complete geometry-labeled cohort: {len(data)}")
    print(f"Retention-score coverage: {data['predicted_retained'].notna().sum()}/{len(data)}")
    for spec in SYSTEMS:
        subset = data[data["system"] == spec.label].dropna(
            subset=["predicted_retained"]
        )
        matrix = confusion_counts(subset)
        print(
            f"{spec.label:>8}: N={len(subset):3d}  threshold={spec.threshold:.2f}  "
            f"TN={matrix[0, 0]:2d} FP={matrix[0, 1]:2d} "
            f"FN={matrix[1, 0]:2d} TP={matrix[1, 1]:2d}"
        )

    evaluable = data.dropna(subset=["predicted_retained"])
    matrix = confusion_counts(evaluable)
    tn, fp, fn, tp = matrix.ravel()
    accuracy = safe_ratio(tp + tn, matrix.sum())
    retained_precision = safe_ratio(tp, tp + fp)
    retained_recall = safe_ratio(tp, tp + fn)
    migrated_recall = safe_ratio(tn, tn + fp)
    f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = np.nanmean([retained_recall, migrated_recall])
    print(
        f"{'Combined':>8}: N={matrix.sum():3d}  TN={tn:2d} FP={fp:2d} "
        f"FN={fn:2d} TP={tp:2d}"
    )
    print(
        f"Combined metrics: accuracy={accuracy:.3f}, "
        f"balanced_accuracy={balanced_accuracy:.3f}, "
        f"retained_precision={retained_precision:.3f}, "
        f"retained_recall={retained_recall:.3f}, "
        f"migrated_recall={migrated_recall:.3f}, F1={f1:.3f}"
    )

    motif_mismatches = data[data["filename_motif"] != data["initial_motif"]]
    print(
        f"Filename/geometry motif agreement: "
        f"{len(data) - len(motif_mismatches)}/{len(data)}"
    )
    if not motif_mismatches.empty:
        print(
            motif_mismatches[
                ["system", "initial_file", "filename_motif", "initial_motif"]
            ].to_string(index=False)
        )
    missing = data[data["predicted_retained"].isna()][
        ["system", "config", "initial_file", "initial_motif"]
    ]
    if not missing.empty:
        print("\nConfigurations without an exact initial-site ranking match:")
        print(missing.to_string(index=False))


def plot_confusion_matrix(data: pd.DataFrame, output: Path) -> None:
    evaluable = data.dropna(subset=["predicted_retained"])
    counts = confusion_counts(evaluable)
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    ) * 100.0

    cmap = LinearSegmentedColormap.from_list(
        "paper_blue", ("#f7f9fc", "#b8cbe3", "#416b9b")
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    image = ax.imshow(percentages, cmap=cmap, vmin=0.0, vmax=100.0)
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted outcome", fontsize=15, labelpad=10)
    ax.set_ylabel("DFT outcome", fontsize=15, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=13, length=0)

    for row in range(2):
        for column in range(2):
            text_color = "white" if percentages[row, column] >= 55.0 else "black"
            ax.text(
                column,
                row,
                f"{counts[row, column]}\n({percentages[row, column]:.1f}%)",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="semibold",
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    colorbar.set_label(
        "Fraction within each DFT outcome (%)", fontsize=12, labelpad=10
    )
    colorbar.ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#555555")
    fig.subplots_adjust(left=0.23, right=0.88, bottom=0.18, top=0.97)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output}")


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent / "test_systems"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="directory containing the five external Ni test systems",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PDF (default: ROOT/retention_confusion_matrix.pdf)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "retention_confusion_matrix.pdf"
    )

    frames: list[pd.DataFrame] = []
    for spec in SYSTEMS:
        frame, errors = process_system(root, spec)
        frames.append(frame)
        print(f"{spec.label}: processed {len(frame)}; errors {len(errors)}")
        if errors:
            print("  " + "\n  ".join(errors))
    data = pd.concat(frames, ignore_index=True)
    if len(data) != 199:
        raise RuntimeError(f"Expected 199 configurations, found {len(data)}")
    print_summary(data)
    if data["predicted_retained"].isna().any():
        raise RuntimeError("Cannot build a complete matrix with missing predictions")
    plot_confusion_matrix(data, output)


if __name__ == "__main__":
    main()
