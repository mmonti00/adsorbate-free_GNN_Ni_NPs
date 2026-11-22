#!/usr/bin/env python3
"""Build descriptor baselines on the exact M6 train/validation split.

The DFT-relaxed OH*+H* frames provide only the target energy and final anchor
indices.  Motif, GCN_O, GCN_H, and the inter-site centroid distance are then
computed exclusively from the corresponding MACE-relaxed bare Ni structure.
This mirrors the M6 evaluation protocol and avoids leaking relaxed adsorbate
geometry into the descriptor baselines.  Only the final M6 training split is
used for fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ase.geometry import find_mic
from ase.io import iread, read
from ase.neighborlist import neighbor_list
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from periodic_site import site_gcn_pbc, site_centroid_distance_pbc


SEED = 32
NI_CN_CUTOFF = 3.00
NI_O_CUTOFF = 2.20
NI_O_EXTENDED_CUTOFF = 2.30
NI_H_CUTOFF = 2.00
NI_H_EXTENDED_CUTOFF = 2.20
OH_BOND_CUTOFF = 1.20
GCN_CN_MAX = {"T": 12.0, "B": 18.0, "H": 22.0, "F": 26.0}

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
CHECKPOINT_DIR = REPOSITORY_ROOT / "models_checkpoints"
MACE_TRAINING_DIR = REPOSITORY_ROOT / "ML_relaxed_bare_geom" / "MACE" / "training"
MACE_FINE_TUNING_DIR = MACE_TRAINING_DIR / "fine-tuning"

BARE_REFERENCES = {
    "Ni25_conf1": MACE_TRAINING_DIR / "Ni25_MACE_opt.xyz",
    "Ni30_conf1": MACE_TRAINING_DIR / "Ni30_MACE_opt.xyz",
    "Ni35_conf1": MACE_TRAINING_DIR / "Ni35_MACE_opt.xyz",
    "Ni40_conf1": MACE_TRAINING_DIR / "Ni40_MACE_opt.xyz",
    "Ni45_conf1": MACE_TRAINING_DIR / "Ni45_MACE_opt.xyz",
    "Ni47_conf1": MACE_FINE_TUNING_DIR / "Ni47c1_MACE_opt.xyz",
    "Ni47_conf2": MACE_FINE_TUNING_DIR / "Ni47c2_MACE_opt.xyz",
    "Ni50_conf1": MACE_TRAINING_DIR / "Ni50_MACE_opt.xyz",
    "Ni111_conf1": MACE_FINE_TUNING_DIR / "Ni111_2x2_MACE_opt.xyz",
    "Ni111_conf2": MACE_FINE_TUNING_DIR / "Ni111_3x3_MACE_opt.xyz",
    "Ni100_conf1": MACE_FINE_TUNING_DIR / "Ni100_3x3_MACE_opt.xyz",
    "Ni100_conf2": MACE_FINE_TUNING_DIR / "Ni100_4x3_MACE_opt.xyz",
}


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    split: str
    domain: str
    weight: float
    priority: int


# The released M6 train/validation structures; nanoclusters and slabs are
# separated by their periodic boundary conditions.
SPLIT_FILES = (
    ("train", CHECKPOINT_DIR / "M6_energy_model_all_train_data.extxyz"),
    ("validation", CHECKPOINT_DIR / "M6_energy_model_all_valid_data.extxyz"),
)
SOURCES = tuple(
    Source(f"M6_{'train' if split == 'train' else 'val'}_{domain}", path, split, domain,
           1.0 if domain == "NP" else 4.0, 3)
    for split, path in SPLIT_FILES
    for domain in ("NP", "SLAB")
)


def distance_pbc(a: np.ndarray, b: np.ndarray, cell, pbc) -> float:
    vector = np.asarray(a, float) - np.asarray(b, float)
    if np.any(pbc):
        vector, _ = find_mic(vector, cell=cell, pbc=pbc)
    return float(np.linalg.norm(vector))


def classify_site(n_anchor: int) -> str:
    if n_anchor <= 1:
        return "T"
    if n_anchor == 2:
        return "B"
    if n_anchor == 3:
        return "H"
    return "F"


def find_adsorbates(atoms) -> tuple[int, int]:
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    o_indices = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    h_indices = [i for i, symbol in enumerate(symbols) if symbol == "H"]
    if len(o_indices) != 1 or len(h_indices) != 2:
        raise ValueError(f"Expected one O and two H atoms; found O={len(o_indices)}, H={len(h_indices)}")
    o_idx = o_indices[0]
    h_by_distance = sorted(
        h_indices,
        key=lambda idx: distance_pbc(positions[idx], positions[o_idx], atoms.cell, atoms.pbc),
    )
    return o_idx, h_by_distance[-1]


def extract_anchors(
    atoms,
    adsorbate_idx: int,
    metal_indices: list[int],
    short_cutoff: float,
    extended_cutoff: float,
) -> tuple[list[int], str]:
    hits = []
    for local_idx, global_idx in enumerate(metal_indices):
        distance = distance_pbc(
            atoms.positions[global_idx], atoms.positions[adsorbate_idx], atoms.cell, atoms.pbc
        )
        if distance < extended_cutoff:
            hits.append((distance, local_idx))
    hits.sort()
    short = [(distance, idx) for distance, idx in hits if distance < short_cutoff]
    if len(short) >= 2 and len(hits) == 4:
        selected = hits[:4]
    else:
        selected = short[:3]
    anchors = [idx for _, idx in selected]
    if not anchors:
        raise ValueError("No adsorption anchors found")
    return anchors, classify_site(len(anchors))


def periodic_coordination(bare, cell, pbc) -> tuple[list[list[tuple[int, tuple[int, int, int]]]], np.ndarray]:
    work = bare.copy()
    work.set_cell(cell)
    work.set_pbc(pbc)
    i_idx, j_idx, shifts = neighbor_list(
        "ijS", work, NI_CN_CUTOFF, self_interaction=False
    )
    neighbors: list[list[tuple[int, tuple[int, int, int]]]] = [list() for _ in range(len(work))]
    for i, j, shift in zip(i_idx, j_idx, shifts):
        neighbors[int(i)].append((int(j), tuple(int(x) for x in shift)))
    cn = np.asarray([len(items) for items in neighbors], dtype=float)
    return neighbors, cn


def infer_group(atoms) -> str:
    keys = list(BARE_REFERENCES)
    for metadata_key in ("bare_label", "initial_label", "group_label"):
        value = atoms.info.get(metadata_key)
        if value is not None and str(value) in BARE_REFERENCES:
            return str(value)
    label = str(atoms.info.get("config_label", atoms.info.get("source_label", "")))
    for key in sorted(keys, key=len, reverse=True):
        if label.startswith(key):
            return key
    raise ValueError(f"Cannot map configuration label {label!r} to a bare Ni reference")


def get_energy(atoms) -> float:
    if atoms.calc is not None and "energy" in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results["energy"])
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    return float(atoms.get_potential_energy())


def process_frame(atoms, source: Source, source_index: int, bare_cache: dict[str, object]) -> dict[str, object]:
    symbols = atoms.get_chemical_symbols()
    metal_indices = [i for i, symbol in enumerate(symbols) if symbol == "Ni"]
    group = infer_group(atoms)
    bare = bare_cache[group]
    if len(bare) != len(metal_indices):
        raise ValueError(f"{group}: bare reference has {len(bare)} Ni atoms, frame has {len(metal_indices)}")

    o_idx, h_ads_idx = find_adsorbates(atoms)
    o_anchor, o_site = extract_anchors(
        atoms, o_idx, metal_indices, NI_O_CUTOFF, NI_O_EXTENDED_CUTOFF
    )
    h_anchor, h_site = extract_anchors(
        atoms, h_ads_idx, metal_indices, NI_H_CUTOFF, NI_H_EXTENDED_CUTOFF
    )

    cell = atoms.cell
    pbc = atoms.pbc
    bare_positions = bare.positions
    neighbors, cn = periodic_coordination(bare, cell, pbc)
    gcn_o, _ = site_gcn_pbc(o_anchor, o_site, neighbors, cn, bare_positions, cell, pbc)
    anchor_cn_o = float(np.mean(cn[o_anchor]))
    gcn_h, _ = site_gcn_pbc(h_anchor, h_site, neighbors, cn, bare_positions, cell, pbc)
    anchor_cn_h = float(np.mean(cn[h_anchor]))
    d_oh = site_centroid_distance_pbc(bare_positions, o_anchor, h_anchor, cell, pbc)
    motif = o_site + h_site
    config_label = str(atoms.info.get("config_label", atoms.info.get("source_label", "")))
    input_key = (
        f"{source.domain}|{group}|O:{','.join(map(str, sorted(o_anchor)))}|"
        f"H:{','.join(map(str, sorted(h_anchor)))}"
    )
    identity_key = f"{source.domain}|{group}|{config_label}|{input_key}"
    return {
        "split": source.split,
        "source": source.name,
        "source_priority": source.priority,
        "source_index": source_index,
        "domain": source.domain,
        "system": group,
        "config_label": config_label,
        "n_metal": len(metal_indices),
        "motif": motif,
        "GCN_O": gcn_o,
        "GCN_H": gcn_h,
        "d_OH_A": d_oh,
        "anchor_CN_O": anchor_cn_o,
        "anchor_CN_H": anchor_cn_h,
        "Eads_DFT_eV": get_energy(atoms),
        "sample_weight": source.weight,
        "o_anchor_idx": ";".join(map(str, sorted(o_anchor))),
        "h_anchor_idx": ";".join(map(str, sorted(h_anchor))),
        "shared_anchor_count": len(set(o_anchor) & set(h_anchor)),
        "input_key": input_key,
        "identity_key": identity_key,
    }


def choose_training_row(rows: list[dict[str, object]]) -> dict[str, object]:
    # Keep one row per repeated labelled configuration.
    return max(rows, key=lambda row: (int(row["source_priority"]), -int(row["source_index"]))).copy()


def deduplicate_training(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        # Remove repeated copies of the same labelled configuration. Do not
        # collapse different labels or symmetry-related anchor choices merely
        # because the baseline descriptors coincide.
        grouped[str(row["identity_key"])].append(row)
    kept = []
    audit = []
    for key, variants in grouped.items():
        selected = choose_training_row(variants)
        energies = [float(row["Eads_DFT_eV"]) for row in variants]
        selected["duplicate_count"] = len(variants)
        selected["duplicate_energy_spread_eV"] = max(energies) - min(energies)
        kept.append(selected)
        if len(variants) > 1:
            audit.append({
                "identity_key": key,
                "input_key": selected["input_key"],
                "n_rows": len(variants),
                "energy_min_eV": min(energies),
                "energy_max_eV": max(energies),
                "energy_spread_eV": max(energies) - min(energies),
                "sources": sorted({str(row["source"]) for row in variants}),
                "config_labels": sorted({str(row["config_label"]) for row in variants}),
                "kept_source": selected["source"],
                "kept_config_label": selected["config_label"],
            })
    kept.sort(key=lambda row: (str(row["domain"]), str(row["system"]), str(row["input_key"])))
    return kept, audit


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer CSV fields for empty output {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "reproduced_results" / "baseline_dataset.csv")
    parser.add_argument("--audit", type=Path, default=HERE / "reproduced_results" / "baseline_dataset_audit.json")
    parser.add_argument("--duplicates", type=Path, default=HERE / "reproduced_results" / "baseline_duplicate_inputs.csv")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in BARE_REFERENCES.values() if not path.exists()]
    missing.extend(str(source.path) for source in SOURCES if not source.path.exists())
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    bare_cache = {label: read(path) for label, path in BARE_REFERENCES.items()}
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    source_counts = Counter()
    failures = []
    for source in SOURCES:
        frames = [atoms for atoms in iread(source.path, index=":")
                  if bool(np.any(atoms.pbc)) == (source.domain == "SLAB")]
        for source_index, atoms in enumerate(frames):
            try:
                row = process_frame(atoms, source, source_index, bare_cache)
            except Exception as exc:
                failures.append({"source": source.name, "source_index": source_index, "error": str(exc)})
                continue
            by_split[source.split].append(row)
            source_counts[source.name] += 1

    if failures:
        preview = "\n".join(str(item) for item in failures[:10])
        raise RuntimeError(f"Descriptor extraction failed for {len(failures)} frames:\n{preview}")

    train_rows, duplicate_audit = deduplicate_training(by_split["train"])
    train_keys = {str(row["input_key"]) for row in train_rows}

    validation_rows = []
    validation_overlap = []
    seen_validation_ids = Counter()
    for row in by_split["validation"]:
        key = str(row["input_key"])
        row = row.copy()
        row["duplicate_count"] = 1
        row["duplicate_energy_spread_eV"] = 0.0
        row["training_input_overlap"] = key in train_keys
        if key in train_keys:
            validation_overlap.append(row)
        # Keep every M6 validation observation for like-for-like reporting.
        seen_validation_ids[key] += 1
        row["validation_input_multiplicity"] = seen_validation_ids[key]
        validation_rows.append(row)

    for row in train_rows:
        row["training_input_overlap"] = False
        row["validation_input_multiplicity"] = 0

    rows = train_rows + validation_rows
    for index, row in enumerate(rows):
        row["row_id"] = index

    fields = [
        "row_id", "split", "source", "source_index", "domain", "system", "config_label",
        "n_metal", "motif", "GCN_O", "GCN_H", "d_OH_A", "anchor_CN_O", "anchor_CN_H",
        "Eads_DFT_eV", "sample_weight", "o_anchor_idx", "h_anchor_idx", "shared_anchor_count",
        "input_key", "duplicate_count", "duplicate_energy_spread_eV", "training_input_overlap",
        "validation_input_multiplicity", "identity_key",
    ]
    write_csv(args.output, rows, fields)
    if duplicate_audit:
        write_csv(args.duplicates, duplicate_audit)
    else:
        write_csv(
            args.duplicates,
            [],
            ["identity_key", "input_key", "n_rows", "energy_min_eV", "energy_max_eV", "energy_spread_eV", "sources", "config_labels", "kept_source", "kept_config_label"],
        )

    audit = {
        "seed": SEED,
        "protocol": {
            "anchors": "DFT-relaxed adsorbate frames, hybrid 2.20/2.30 A O and 2.00/2.20 A H definitions",
            "descriptors": "MACE-relaxed bare Ni structures only",
            "gcn_cutoff_A": NI_CN_CUTOFF,
            "periodic_handling": "site anchors placed in one compact set of periodic images; GCN neighbours counted as (atom, image) pairs relative to those images, excluding the anchors; minimum-image distance between compact site centroids",
            "train_definition": "exact M6 nanoparticle and slab training split; trained from scratch with no M5 merge",
            "validation_definition": "original M6 NP/slab validation observations",
        },
        "repository_files": {
            "train_structures": "../models_checkpoints/M6_energy_model_all_train_data.extxyz",
            "validation_structures": "../models_checkpoints/M6_energy_model_all_valid_data.extxyz",
            "bare_geometries": "../ML_relaxed_bare_geom/MACE/training",
        },
        "raw_source_counts": dict(source_counts),
        "n_train_raw": len(by_split["train"]),
        "n_train_unique_configurations": len(train_rows),
        "n_unique_train_inputs": len(train_keys),
        "n_duplicate_identity_groups": len(duplicate_audit),
        "max_duplicate_energy_spread_eV": max(
            (float(item["energy_spread_eV"]) for item in duplicate_audit), default=0.0
        ),
        "n_validation": len(validation_rows),
        "n_validation_inputs_seen_in_training": len(validation_overlap),
        "n_unique_validation_inputs_seen_in_training": len({row["input_key"] for row in validation_overlap}),
        "train_by_domain": dict(Counter(str(row["domain"]) for row in train_rows)),
        "validation_by_domain": dict(Counter(str(row["domain"]) for row in validation_rows)),
        "train_by_system": dict(Counter(str(row["system"]) for row in train_rows)),
        "validation_by_system": dict(Counter(str(row["system"]) for row in validation_rows)),
        "train_by_motif": dict(Counter(str(row["motif"]) for row in train_rows)),
        "validation_by_motif": dict(Counter(str(row["motif"]) for row in validation_rows)),
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.audit}")
    print(f"Wrote {args.duplicates}")


if __name__ == "__main__":
    main()
