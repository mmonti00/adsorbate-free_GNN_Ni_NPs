"""Fine-tune the M5 model on nanoparticles and periodic slabs to produce M6.

The adsorbate-containing extxyz datasets provide DFT adsorption-energy targets
and relaxed OH*/H* geometries from which the Ni anchors are assigned. Input
graphs are built from the corresponding MACE-relaxed bare Ni structures listed
in ``INITIAL_NI_FILES``. Nanoparticle graphs are finite, while slab graphs use
explicit periodic images generated from the lattice and PBC stored in the bare
extxyz files.

To reproduce the paper, use the split supplied in ``paper_splits/``; this is
the default ``FIXED_SPLIT_JSON`` setting. The fixed manifest takes precedence
over the adsorption-state filter so that its stored membership is reproduced
exactly. Setting ``FIXED_SPLIT_JSON = None`` instead filters the target
adsorption state and creates a new seeded stratified train/validation split.

Seeds are set for Python, NumPy, PyTorch, and data-loader shuffling. Exact
bitwise reproducibility is nevertheless not guaranteed for every CUDA/PyG
operation or across different hardware and software versions, so repeated runs
can give slightly different fitted parameters and validation metrics.

The script writes the split datasets, best checkpoint, and a JSON summary. Run
it from its own directory so that the relative paths below resolve as provided.
"""

import copy
import json
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.io import read, write
from ase.geometry import find_mic
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, MessagePassing

# ============================================================
# INPUTS, OUTPUTS, AND MODEL SETTINGS
# ============================================================

# Adsorbate-containing configurations and adsorption-energy targets. Set the
# slab path to None for nanoparticle-only training.
NP_EXTXYZ_FILE = "../datasets/dataset_NPs_M6_energy_model.extxyz"
SLAB_EXTXYZ_FILE = "../datasets/dataset_SLABs_M6_energy_model.extxyz"

# To reproduce the paper, use the supplied split. Set to None for a new seeded
# stratified split.
FIXED_SPLIT_JSON = "paper_splits/M6_energy_model_paper_split.json"

# Generated split files.
TRAIN_EXTXYZ_OUT = "M6_energy_model_all_train_data.extxyz"
VAL_EXTXYZ_OUT = "M6_energy_model_all_valid_data.extxyz"
TRAIN_NP_EXTXYZ_OUT = "M6_energy_model_NPs_train_data.extxyz"
VAL_NP_EXTXYZ_OUT = "M6_energy_model_NPs_valid_data.extxyz"
TRAIN_SLAB_EXTXYZ_OUT = "M6_energy_model_SLABs_train_data.extxyz"
VAL_SLAB_EXTXYZ_OUT = "M6_energy_model_SLABs_valid_data.extxyz"

# Generated checkpoint and dataset summary.
MODEL_OUT = "M6_energy_model.pt"
STATS_OUT = "M6_energy_model_dataset_stats.json"

# Initialize M6 from the nanoparticle-only M5 energy model. Set FINE_TUNE to
# False to train the same architecture from scratch.
FINE_TUNE = True
PRETRAINED_MODEL_PATH = "../models_checkpoints/M5_energy_model.pt"

# When True, freeze the three message-passing layers and optimize only the
# energy head.
FREEZE_MESSAGE_PASSING = False

# MACE-relaxed bare structures used to construct the input graphs. Each dataset
# frame should store one exact dictionary key in bare_label, initial_label, or
# group_label. If absent, config_label/source_label is matched by prefix.
INITIAL_NI_FILES = {
    # Finite nanoparticles: ordinary XYZ is sufficient because PBC are off.
    "Ni25_conf1": "../ML_relaxed_bare_geom/MACE/training/Ni25_MACE_opt.xyz",
    "Ni30_conf1": "../ML_relaxed_bare_geom/MACE/training/Ni30_MACE_opt.xyz",
    "Ni35_conf1": "../ML_relaxed_bare_geom/MACE/training/Ni35_MACE_opt.xyz",
    "Ni40_conf1": "../ML_relaxed_bare_geom/MACE/training/Ni40_MACE_opt.xyz",
    "Ni45_conf1": "../ML_relaxed_bare_geom/MACE/training/Ni45_MACE_opt.xyz",
    "Ni47_conf1": "../ML_relaxed_bare_geom/MACE/training/fine-tuning/Ni47c1_MACE_opt.xyz",
    "Ni47_conf2": "../ML_relaxed_bare_geom/MACE/training/fine-tuning/Ni47c2_MACE_opt.xyz",
    "Ni50_conf1": "../ML_relaxed_bare_geom/MACE/training/Ni50_MACE_opt.xyz",
    # Periodic slabs: extxyz preserves the lattice and PBC used to build the
    # explicit periodic-image graph.
    "Ni111_conf1": "../ML_relaxed_bare_geom/MACE/training/fine-tuning/Ni111_2x2_MACE_opt.extxyz",
    "Ni111_conf2": "../ML_relaxed_bare_geom/MACE/training/fine-tuning/Ni111_3x3_MACE_opt.extxyz",
    "Ni100_conf1": "../ML_relaxed_bare_geom/MACE/training/fine-tuning/Ni100_3x3_MACE_opt.extxyz",
    "Ni100_conf2": "../ML_relaxed_bare_geom/MACE/training/fine-tuning/Ni100_4x3_MACE_opt.extxyz",
}

TARGET_ENERGY_LABEL = "energy"

SEED = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Adsorbate and anchor definitions ---

# Standard T/B/H O sites use Ni-O contacts below 2.20 Å. An F site requires
# at least two such contacts and exactly four contacts below 2.30 Å.
NI_O_CUTOFF = 2.20
NI_O_FOLDED_CUTOFF = 2.30
O_FOLDED_N_SHORT = 2
O_FOLDED_N_TOTAL = 4

# Standard T/B/H H sites use Ni-H contacts below 2.00 Å. An F site requires
# at least two such contacts and exactly four contacts below 2.20 Å.
NI_H_CUTOFF = 2.00
NI_H_FOLDED_CUTOFF = 2.20
H_FOLDED_N_SHORT = 2
H_FOLDED_N_TOTAL = 4

# The H atom bonded to O is assigned as the hydroxyl H; the other is H*.
OH_BOND_CUTOFF = 1.20

# F sites retain four anchors; standard sites retain at most three.
MAX_STANDARD_O_ANCHORS = 3
MAX_STANDARD_H_ANCHORS = 3

# --- Local graph built from the bare Ni structure ---
# Include Ni atoms within this distance of any O/H anchor.
LOCAL_CUTOFF = 5.5

# Connect Ni nodes separated by at most this distance.
EDGE_CUTOFF = 4.0

# A value overrides the global cutoff for that domain; None uses the global
# value above.
LOCAL_CUTOFF_BY_DOMAIN = {
    "NP": None,
    "SLAB": 6.0,
}
EDGE_CUTOFF_BY_DOMAIN = {
    "NP": None,
    "SLAB": 4.5,
}

# The dataset and bare extxyz cells differ only at the printed-coordinate
# precision level. Larger discrepancies indicate a mismatched reference file.
CELL_MATCH_ATOL = 1.0e-5


def get_local_cutoff(domain_label: str) -> float:
    v = LOCAL_CUTOFF_BY_DOMAIN.get(domain_label)
    return float(v) if v is not None else LOCAL_CUTOFF


def get_edge_cutoff(domain_label: str) -> float:
    v = EDGE_CUTOFF_BY_DOMAIN.get(domain_label)
    return float(v) if v is not None else EDGE_CUTOFF

# --- distance basis expansion ---
N_RBF = 16
RBF_DMIN = 0.0
RBF_DMAX = 5.0
RBF_GAMMA = 10.0

# --- training ---
BATCH_SIZE = 32
LR = 3e-4 if FINE_TUNE else 8e-4
WEIGHT_DECAY = 1e-6
EPOCHS = 100 if FINE_TUNE else 200
VAL_FRACTION_NP = 0.20
VAL_FRACTION_SLAB = 0.20
EARLY_STOPPING_PATIENCE = 25 if FINE_TUNE else 25
HIDDEN_DIM = 128

# Per-sample regression weights balance the smaller slab dataset.
NP_SAMPLE_WEIGHT = 1.0
SLAB_SAMPLE_WEIGHT = 4.0

# Energy-regression objective.
REG_LOSS_WEIGHT = 1.0

# Length scale and anchor emphasis used by the O- and H-focused pooling heads.
POOL_SIGMA = 1.5
ANCHOR_BOOST = 2.0

# Site types form two-letter motifs: the O site first, followed by the H* site.
SITE_TYPES = ("T", "B", "H", "F")
MOTIF_TO_ID = {
    f"{o_site}{h_site}": idx
    for idx, (o_site, h_site) in enumerate(
        (o_site, h_site)
        for o_site in SITE_TYPES
        for h_site in SITE_TYPES
    )
}
ID_TO_MOTIF = {v: k for k, v in MOTIF_TO_ID.items()}

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# PBC HELPERS
# ============================================================

def mic_vectors(vectors: np.ndarray, cell, pbc) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if not np.any(pbc):
        return vectors
    v_mic, _ = find_mic(vectors, cell=cell, pbc=pbc)
    return np.asarray(v_mic)

def mic_distances(vectors: np.ndarray, cell, pbc) -> np.ndarray:
    return np.linalg.norm(mic_vectors(vectors, cell, pbc), axis=-1)

def distance_pbc(pos_a: np.ndarray, pos_b: np.ndarray, cell, pbc) -> float:
    return float(mic_distances(np.asarray(pos_a) - np.asarray(pos_b), cell, pbc))

def unwrap_positions_around_reference(
    positions: np.ndarray,
    indices: List[int],
    cell,
    pbc,
    ref_idx: Optional[int] = None,
) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, 3), dtype=float)
    if ref_idx is None:
        ref_idx = indices[0]
    ref_pos = positions[ref_idx]
    out = []
    for idx in indices:
        vec = positions[idx] - ref_pos
        out.append(ref_pos + mic_vectors(vec, cell, pbc))
    return np.asarray(out, dtype=float)

def center_of_indices_pbc(positions: np.ndarray, indices: List[int], cell, pbc) -> np.ndarray:
    return unwrap_positions_around_reference(positions, indices, cell, pbc).mean(axis=0)

# ============================================================
# HELPERS
# ============================================================

def get_target_energy(atoms) -> float:
    """Return the adsorption-energy target stored in an extxyz frame."""
    if atoms.calc is not None and hasattr(atoms.calc, "results"):
        if TARGET_ENERGY_LABEL in atoms.calc.results:
            return float(atoms.calc.results[TARGET_ENERGY_LABEL])
    if TARGET_ENERGY_LABEL in atoms.info:
        return float(atoms.info[TARGET_ENERGY_LABEL])
    try:
        return float(atoms.get_potential_energy())
    except Exception as e:
        raise ValueError(
            f"Could not find target {TARGET_ENERGY_LABEL!r}. "
            f"Available atoms.info keys: {list(atoms.info.keys())}"
        ) from e

def gaussian_distance_expansion(
    distances: np.ndarray,
    dmin: float = RBF_DMIN,
    dmax: float = RBF_DMAX,
    n_kernels: int = N_RBF,
    gamma: float = RBF_GAMMA,
) -> np.ndarray:
    centers = np.linspace(dmin, dmax, n_kernels)
    diff = distances[:, None] - centers[None, :]
    return np.exp(-gamma * diff ** 2)

def classify_site(n_neighbors: int) -> str:
    """Map the number of extracted Ni anchors to a site label.

    T = top / onefold
    B = bridge / twofold
    H = hollow / threefold
    F = folded or fourfold / four anchors
    """
    if n_neighbors <= 1:
        return "T"
    if n_neighbors == 2:
        return "B"
    if n_neighbors == 3:
        return "H"
    return "F"

def get_symbols_positions(atoms) -> Tuple[List[str], np.ndarray]:
    return atoms.get_chemical_symbols(), atoms.get_positions()

def count_ni_atoms(symbols: List[str]) -> int:
    return sum(1 for s in symbols if s == "Ni")

def find_adsorbate_indices(symbols: List[str], positions: np.ndarray, cell, pbc) -> Tuple[int, int, int]:
    """Return the O, hydroxyl-H, and independently adsorbed-H indices.

    Distances use the minimum-image convention for periodic frames. Each frame
    must contain exactly one H within ``OH_BOND_CUTOFF`` of O. This rejects both
    dissociated O*+H*+H* and molecular H2O-like configurations from the target
    OH*+H* state.
    """
    o_indices = [i for i, s in enumerate(symbols) if s == "O"]
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]

    if len(o_indices) != 1:
        raise ValueError(f"Expected exactly 1 O, found {len(o_indices)}")
    if len(h_indices) != 2:
        raise ValueError(f"Expected exactly 2 H, found {len(h_indices)}")

    o_idx = o_indices[0]
    o_pos = positions[o_idx]

    oh_like = []
    for h_idx in h_indices:
        d = distance_pbc(positions[h_idx], o_pos, cell, pbc)
        if d < OH_BOND_CUTOFF:
            oh_like.append((h_idx, d))

    if len(oh_like) == 1:
        h_oh_idx = oh_like[0][0]
        h_ads_idx = [h for h in h_indices if h != h_oh_idx][0]
        return o_idx, h_oh_idx, h_ads_idx

    distances = [distance_pbc(positions[h_idx], o_pos, cell, pbc) for h_idx in h_indices]

    # A fixed manifest reproduces the preprocessing used to create that split.
    # If an explicitly listed frame has no unique O-H assignment, retain the
    # historical convention: the H closest to O is OH and the other is H*.
    if FIXED_SPLIT_JSON is not None:
        ordered = sorted(zip(h_indices, distances), key=lambda item: item[1])
        print(
            "  Fixed-split compatibility assignment: no unique O-H bond for "
            f"distances={distances}; assigning closest H as hydroxyl H."
        )
        return o_idx, ordered[0][0], ordered[1][0]

    raise ValueError(
        f"Expected exactly one O-H distance below {OH_BOND_CUTOFF:.2f} Å, "
        f"found {len(oh_like)}; distances={distances}"
    )


def filter_target_adsorption_state(frames, domain_label: str):
    """Keep only frames representing the target OH*+H* adsorption state."""
    if FIXED_SPLIT_JSON is not None:
        print(
            f"Fixed split active: retaining all {len(frames)} {domain_label} "
            "frames before graph construction."
        )
        return list(frames)

    kept = []
    rejected = []

    for input_idx, atoms in enumerate(frames):
        symbols, positions = get_symbols_positions(atoms)
        try:
            find_adsorbate_indices(symbols, positions, atoms.cell, atoms.pbc)
        except ValueError as exc:
            rejected.append(
                (input_idx, str(atoms.info.get("config_label", "")), str(exc))
            )
        else:
            kept.append(atoms)

    print(
        f"Retained {len(kept)} {domain_label} OH*+H* frames; "
        f"discarded {len(rejected)} invalid frames."
    )
    for input_idx, config_label, reason in rejected:
        print(f"  Discarded {domain_label} input frame {input_idx} ({config_label}): {reason}")

    return kept

def extract_o_anchor_indices_hybrid(
    ni_indices: List[int],
    positions: np.ndarray,
    o_idx: int,
    cell,
    pbc,
) -> Tuple[List[int], str, Dict[str, object]]:
    """Assign the PBC-aware O anchors and site type.

    Standard T/B/H sites use at most three contacts below ``NI_O_CUTOFF``.
    An F site requires at least ``O_FOLDED_N_SHORT`` short contacts and exactly
    ``O_FOLDED_N_TOTAL`` contacts below ``NI_O_FOLDED_CUTOFF``; all four are
    then retained.
    """
    o_pos = positions[o_idx]
    extended_hits = []

    for ni_idx in ni_indices:
        d = distance_pbc(positions[ni_idx], o_pos, cell, pbc)
        if d < NI_O_FOLDED_CUTOFF:
            extended_hits.append((ni_idx, d))

    extended_hits.sort(key=lambda item: item[1])
    short_hits = [
        (idx, d)
        for idx, d in extended_hits
        if d < NI_O_CUTOFF
    ]

    is_folded = (
        len(short_hits) >= O_FOLDED_N_SHORT
        and len(extended_hits) == O_FOLDED_N_TOTAL
    )

    if is_folded:
        selected_hits = extended_hits[:O_FOLDED_N_TOTAL]
        o_site = "F"
        rule = "folded_at_least_2short_four_total"
    else:
        selected_hits = short_hits[:MAX_STANDARD_O_ANCHORS]
        o_site = classify_site(len(selected_hits))
        rule = "standard_short_cutoff"

    metadata = {
        "o_anchor_rule": rule,
        "o_short_contact_count": int(len(short_hits)),
        "o_extended_contact_count": int(len(extended_hits)),
        "o_short_contact_distances": [float(d) for _, d in short_hits],
        "o_extended_contact_distances": [float(d) for _, d in extended_hits],
    }
    return [idx for idx, _ in selected_hits], o_site, metadata

def extract_h_anchor_indices_hybrid(
    ni_indices: List[int],
    positions: np.ndarray,
    h_ads_idx: int,
    cell,
    pbc,
) -> Tuple[List[int], str, Dict[str, object]]:
    """Assign the PBC-aware H* anchors and site type.

    Standard T/B/H sites use at most three contacts below ``NI_H_CUTOFF``.
    An F site requires at least ``H_FOLDED_N_SHORT`` short contacts and exactly
    ``H_FOLDED_N_TOTAL`` contacts below ``NI_H_FOLDED_CUTOFF``; all four are
    then retained.
    """
    h_pos = positions[h_ads_idx]
    extended_hits = []

    for ni_idx in ni_indices:
        d = distance_pbc(positions[ni_idx], h_pos, cell, pbc)
        if d < NI_H_FOLDED_CUTOFF:
            extended_hits.append((ni_idx, d))

    extended_hits.sort(key=lambda item: item[1])
    short_hits = [
        (idx, d)
        for idx, d in extended_hits
        if d < NI_H_CUTOFF
    ]

    is_folded = (
        len(short_hits) >= H_FOLDED_N_SHORT
        and len(extended_hits) == H_FOLDED_N_TOTAL
    )

    if is_folded:
        selected_hits = extended_hits[:H_FOLDED_N_TOTAL]
        h_site = "F"
        rule = "folded_at_least_2short_four_total"
    else:
        selected_hits = short_hits[:MAX_STANDARD_H_ANCHORS]
        h_site = classify_site(len(selected_hits))
        rule = "standard_short_cutoff"

    metadata = {
        "h_anchor_rule": rule,
        "h_short_contact_count": int(len(short_hits)),
        "h_extended_contact_count": int(len(extended_hits)),
        "h_short_contact_distances": [float(d) for _, d in short_hits],
        "h_extended_contact_distances": [float(d) for _, d in extended_hits],
    }
    return [idx for idx, _ in selected_hits], h_site, metadata

def select_bare_local_environment(
    ni_indices: List[int],
    positions: np.ndarray,
    anchor_indices: List[int],
    cell,
    pbc,
    cutoff: float = LOCAL_CUTOFF,
) -> List[int]:
    """Select bare Ni atoms within ``cutoff`` of any anchor."""
    if len(anchor_indices) == 0:
        raise ValueError("No anchors found; cannot build local environment.")

    keep = set(anchor_indices)
    anchor_positions = positions[anchor_indices]

    for ni_idx in ni_indices:
        ni_pos = positions[ni_idx]
        dmin = float(np.min(mic_distances(anchor_positions - ni_pos[None, :], cell, pbc)))
        if dmin <= cutoff:
            keep.add(ni_idx)

    return sorted(keep)

def compute_coordination_numbers(
    ni_positions: np.ndarray,
    cell,
    pbc,
    cutoff: float = EDGE_CUTOFF,
) -> np.ndarray:
    """Count neighboring Ni atoms within ``cutoff`` for each local node."""
    n = len(ni_positions)
    cn = np.zeros(n, dtype=np.float32)
    for i in range(n):
        count = 0
        for j in range(n):
            if i == j:
                continue
            d = distance_pbc(ni_positions[j], ni_positions[i], cell, pbc)
            if d <= cutoff:
                count += 1
        cn[i] = float(count)
    return cn

def build_node_features(
    keep_indices: List[int],
    positions: np.ndarray,
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    cell,
    pbc,
    edge_cutoff: float = EDGE_CUTOFF,
) -> np.ndarray:
    """Build the eight node features used by the energy model."""
    local_positions = positions[keep_indices]
    cn = compute_coordination_numbers(local_positions, cell=cell, pbc=pbc, cutoff=edge_cutoff)

    o_set = set(o_anchor_idx)
    h_set = set(h_anchor_idx)
    any_set = o_set | h_set

    o_center = center_of_indices_pbc(positions, o_anchor_idx, cell, pbc)
    h_center = center_of_indices_pbc(positions, h_anchor_idx, cell, pbc)
    any_center = center_of_indices_pbc(positions, sorted(any_set), cell, pbc)

    feats = []
    for local_i, global_idx in enumerate(keep_indices):
        pos = positions[global_idx]
        feats.append([
            1.0,
            1.0 if global_idx in o_set else 0.0,
            1.0 if global_idx in h_set else 0.0,
            1.0 if global_idx in any_set else 0.0,
            distance_pbc(pos, o_center, cell, pbc),
            distance_pbc(pos, h_center, cell, pbc),
            distance_pbc(pos, any_center, cell, pbc),
            float(cn[local_i]),
        ])
    return np.asarray(feats, dtype=np.float32)

def build_edges_and_edge_attr(
    local_positions: np.ndarray,
    cell,
    pbc,
    cutoff: float = EDGE_CUTOFF,
):
    """Build directed Ni-Ni edges and Gaussian distance features."""
    edge_src = []
    edge_dst = []
    distances = []

    n = len(local_positions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = distance_pbc(local_positions[j], local_positions[i], cell, pbc)
            if d <= cutoff:
                edge_src.append(i)
                edge_dst.append(j)
                distances.append(d)

    if len(distances) == 0:
        raise ValueError("No edges found. Try increasing EDGE_CUTOFF.")

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(
        gaussian_distance_expansion(np.asarray(distances, dtype=np.float32)),
        dtype=torch.float,
    )
    return edge_index, edge_attr

# ============================================================
# EXPLICIT PERIODIC-IMAGE GRAPH CONSTRUCTION (SLAB ONLY)
# ============================================================
# Slab nodes are identified by (base Ni index, integer cell shift), so more
# than one periodic image of a base atom may appear in a local graph. Once the
# image cloud is generated, all graph distances are ordinary Euclidean
# distances. This makes the local representation independent of where the
# adsorption site lies relative to a supercell boundary.

def _unwrap_anchor_group(
    indices: List[int],
    positions: np.ndarray,
    cell,
    pbc,
) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[int, np.ndarray]]:
    """Unwrap one adsorbate's anchors relative to its first anchor.

    Returns a cell-shift map and the corresponding Cartesian positions. Each
    anchor is placed using its direct minimum-image vector to the fixed
    reference anchor.
    """
    if not indices:
        return {}, {}
    cell_arr = np.asarray(cell)
    inv_cell = np.linalg.inv(cell_arr)

    ref = indices[0]
    ref_pos = positions[ref]
    canonical_pos: Dict[int, np.ndarray] = {ref: ref_pos}
    shifts: Dict[int, Tuple[int, int, int]] = {ref: (0, 0, 0)}

    for idx in indices[1:]:
        if idx in shifts:
            continue
        raw_vec = positions[idx] - ref_pos
        mic_vec = mic_vectors(raw_vec[None, :], cell, pbc)[0]
        canonical_pos[idx] = ref_pos + mic_vec
        shift_cart = mic_vec - raw_vec
        frac = shift_cart @ inv_cell
        shifts[idx] = tuple(int(round(v)) for v in frac)
    return shifts, canonical_pos


def resolve_anchor_shifts(
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    positions: np.ndarray,
    cell,
    pbc,
) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]]:
    """Legacy anchor-only image resolution used for bare-site inference."""
    cell_arr = np.asarray(cell)
    inv_cell = np.linalg.inv(cell_arr)

    o_shifts, o_canonical = _unwrap_anchor_group(
        o_anchor_idx, positions, cell, pbc
    )
    h_shifts, h_canonical = _unwrap_anchor_group(
        h_anchor_idx, positions, cell, pbc
    )

    if o_canonical and h_canonical:
        best = None  # (distance, rigid_shift_cart_for_whole_h_group)
        for o_pos in o_canonical.values():
            for h_pos in h_canonical.values():
                raw_vec = h_pos - o_pos
                mic_vec = mic_vectors(raw_vec[None, :], cell, pbc)[0]
                d = float(np.linalg.norm(mic_vec))
                if best is None or d < best[0]:
                    best = (d, (o_pos + mic_vec) - h_pos)
        _, rigid_shift_cart = best
        rigid_shift = tuple(int(round(v)) for v in rigid_shift_cart @ inv_cell)
        h_shifts = {
            idx: tuple(a + b for a, b in zip(s, rigid_shift))
            for idx, s in h_shifts.items()
        }

    return o_shifts, h_shifts


def _closest_image_shift(base_pos, target, cell_arr, inv_cell, pbc_arr):
    """Return the integer shift placing ``base_pos`` closest to ``target``.

    Also returns the closest distance and the distance margin to the second-
    closest image. A small margin flags a genuinely ambiguous assignment.
    """
    frac = (target - base_pos) @ inv_cell
    s0 = np.where(pbc_arr, np.round(frac), 0.0).astype(int)
    offsets = [range(-1, 2) if periodic else range(1) for periodic in pbc_arr]
    candidates = []
    for dx in offsets[0]:
        for dy in offsets[1]:
            for dz in offsets[2]:
                shift = s0 + np.array([dx, dy, dz])
                distance = float(
                    np.linalg.norm(base_pos + shift @ cell_arr - target)
                )
                candidates.append((distance, tuple(int(v) for v in shift)))
    candidates.sort(key=lambda item: item[0])
    margin = (
        candidates[1][0] - candidates[0][0]
        if len(candidates) > 1
        else np.inf
    )
    return candidates[0][1], candidates[0][0], margin


def resolve_anchor_shifts_from_adsorbates(
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    positions_ads: np.ndarray,
    o_idx: int,
    h_ads_idx: int,
    positions_bare: np.ndarray,
    cell,
    pbc,
    max_anchor_distance: float = 3.0,
    min_margin: float = 0.3,
) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]]:
    """Resolve bare-Ni anchor images from the actual O and H* positions.

    O anchors use the bare periodic images closest to O. H* is first placed in
    its minimum-image position relative to O, and H anchors use the images
    closest to that target. This remains well defined when two Ni anchors are
    exactly half a lattice vector apart.
    """
    cell_arr = np.asarray(cell)
    inv_cell = np.linalg.inv(cell_arr)
    pbc_arr = np.asarray(pbc, dtype=bool)

    o_target = positions_ads[o_idx]
    h_target = o_target + mic_vectors(
        (positions_ads[h_ads_idx] - o_target)[None, :], cell, pbc
    )[0]

    def _resolve(indices, target, label):
        shifts = {}
        for base_idx in indices:
            shift, distance, margin = _closest_image_shift(
                positions_bare[base_idx], target, cell_arr, inv_cell, pbc_arr
            )
            if distance > max_anchor_distance:
                raise ValueError(
                    f"{label} anchor {base_idx}: closest bare image is "
                    f"{distance:.2f} A from the adsorbate; Ni ordering or bare "
                    "reference probably mismatched."
                )
            if margin < min_margin:
                print(
                    f"  Warning: ambiguous {label} anchor image for Ni "
                    f"{base_idx} (d={distance:.2f} A, margin={margin:.2f} A)."
                )
            shifts[base_idx] = shift
        return shifts

    return _resolve(o_anchor_idx, o_target, "O"), _resolve(
        h_anchor_idx, h_target, "H*"
    )


def build_slab_local_graph(
    ni_indices: List[int],
    positions_bare: np.ndarray,
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    anchor_positions: np.ndarray,
    cell,
    pbc,
    local_cutoff: float,
    edge_cutoff: float,
    o_shifts: Optional[Dict[int, Tuple[int, int, int]]] = None,
    h_shifts: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> Tuple[List[Tuple[int, Tuple[int, int, int], np.ndarray]], np.ndarray, np.ndarray, List[int]]:
    """Build an explicit periodic-image graph for one slab adsorption site.

    Returns the image identities, eight-column node-feature array, Cartesian
    image positions, and base Ni indices. Base indices can repeat when several
    periodic copies fall inside the local crop.
    """
    if o_shifts is None or h_shifts is None:
        o_shifts, h_shifts = resolve_anchor_shifts(
            o_anchor_idx, h_anchor_idx, anchor_positions, cell, pbc
        )

    cell_arr = np.asarray(cell)

    def _cart_of(b, shift):
        return positions_bare[b] + np.array(shift) @ cell_arr

    # A shared base atom remains two nodes when O and H* require different
    # periodic images, and collapses to one node when their shifts agree.
    o_anchor_images = {(b, o_shifts[b]) for b in o_anchor_idx}
    h_anchor_images = {(b, h_shifts[b]) for b in h_anchor_idx}
    any_anchor_images = o_anchor_images | h_anchor_images

    o_center = np.mean([_cart_of(b, s) for b, s in o_anchor_images], axis=0)
    h_center = np.mean([_cart_of(b, s) for b, s in h_anchor_images], axis=0)
    any_center = np.mean([_cart_of(b, s) for b, s in any_anchor_images], axis=0)

    anchor_cart = np.array([_cart_of(b, s) for b, s in any_anchor_images])

    pbc_arr = np.asarray(pbc, dtype=bool)
    inv_cell = np.linalg.inv(cell_arr)
    pads = []
    for d in range(3):
        if pbc_arr[d]:
            pads.append(
                int(np.ceil(local_cutoff / np.linalg.norm(cell_arr[d]))) + 1
            )
        else:
            pads.append(0)

    images: List[Tuple[int, Tuple[int, int, int], np.ndarray]] = []
    for b in ni_indices:
        base_pos = positions_bare[b]
        # Center this atom's image search on the images closest to the resolved
        # anchors. This is invariant when individual input atoms are wrapped by
        # different lattice vectors; a single global shift range is not.
        centers = []
        for target in anchor_cart:
            frac = (target - base_pos) @ inv_cell
            centers.append(np.where(pbc_arr, np.round(frac), 0.0).astype(int))
        centers = np.asarray(centers, dtype=int)
        ranges = []
        for d in range(3):
            if pbc_arr[d]:
                lo = int(centers[:, d].min()) - pads[d]
                hi = int(centers[:, d].max()) + pads[d]
                ranges.append(range(lo, hi + 1))
            else:
                ranges.append(range(1))
        for sx in ranges[0]:
            for sy in ranges[1]:
                for sz in ranges[2]:
                    shift = (sx, sy, sz)
                    cart = base_pos + np.array(shift) @ cell_arr
                    dmin = float(np.min(np.linalg.norm(anchor_cart - cart[None, :], axis=1)))
                    if dmin <= local_cutoff:
                        images.append((b, shift, cart))

    if not images:
        raise ValueError("No Ni images found within local_cutoff of the anchor cluster.")

    local_positions = np.array([im[2] for im in images], dtype=float)

    # Use the cropped explicit-image graph for CN, matching the feature
    # convention and scale used by the nanoparticle model.
    cn_local = compute_coordination_numbers(
        local_positions, cell=cell_arr, pbc=[False, False, False], cutoff=edge_cutoff,
    )

    feats = []
    for (b, shift, cart), cn_i in zip(images, cn_local):
        key = (b, shift)
        feats.append([
            1.0,
            1.0 if key in o_anchor_images else 0.0,
            1.0 if key in h_anchor_images else 0.0,
            1.0 if key in any_anchor_images else 0.0,
            float(np.linalg.norm(cart - o_center)),
            float(np.linalg.norm(cart - h_center)),
            float(np.linalg.norm(cart - any_center)),
            float(cn_i),
        ])
    x = np.asarray(feats, dtype=np.float32)

    keep_ni_debug = [im[0] for im in images]
    return images, x, local_positions, keep_ni_debug


def infer_group_label(atoms, available_group_keys: List[str]) -> str:
    """Map one dataset frame to its bare reference structure."""
    # Prefer an explicit extxyz label.
    for key in ["bare_label", "initial_label", "group_label"]:
        if key in atoms.info:
            val = str(atoms.info[key])
            if val in available_group_keys:
                return val

    config_label = str(atoms.info.get("config_label", atoms.info.get("source_label", "")))

    # Fall back to longest-prefix matching; for example,
    # Ni111_conf1_0135 maps to Ni111_conf1.
    for key in sorted(available_group_keys, key=len, reverse=True):
        if config_label.startswith(key):
            return key

    m = re.match(r"(Ni\d+(?:_conf\d+)?)_", config_label)
    if m:
        return m.group(1)
    return "unknown"

def build_group_labels(frames, available_group_keys: List[str]) -> List[str]:
    """Return the bare-reference label for every dataset frame."""
    return [infer_group_label(atoms, available_group_keys) for atoms in frames]

def stratified_split(labels: List[str], val_fraction: float, seed: int = 42):
    """Split indices while preserving each supplied stratum when possible."""
    rng = np.random.default_rng(seed)
    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        label_to_indices[lab].append(idx)

    train_idx = []
    val_idx = []
    for _, idxs in label_to_indices.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_fraction)))
        val_part = idxs[:n_val]
        train_part = idxs[n_val:]
        if len(train_part) == 0:
            train_part = val_part[:1]
            val_part = val_part[1:]
        train_idx.extend(train_part)
        val_idx.extend(val_part)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

def make_combined_strat_labels(domain_labels: List[str], group_labels: List[str], motif_labels: List[str]) -> List[str]:
    """Combine domain, bare-reference, and motif labels for stratification."""
    return [f"{d}__{g}__{m}" for d, g, m in zip(domain_labels, group_labels, motif_labels)]

def summarize_counter(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}

# ============================================================
# DATASET BUILDING
# ============================================================

def load_initial_reference_structures():
    """Load and validate the MACE-relaxed bare Ni reference structures."""
    refs = {}
    for group_label, filename in INITIAL_NI_FILES.items():
        atoms = read(filename)
        symbols = atoms.get_chemical_symbols()
        if any(s != "Ni" for s in symbols):
            raise ValueError(f"{group_label} reference contains non-Ni atoms.")
        refs[group_label] = atoms
    return refs

def process_frame(
    atoms,
    initial_refs,
    group_label: str,
    frame_idx: int,
    domain_label: str,
    sample_weight: float,
):
    """Convert one adsorbate-containing frame into a bare-site graph."""
    symbols, positions = get_symbols_positions(atoms)
    n_ni = count_ni_atoms(symbols)
    e_ads = get_target_energy(atoms)

    cell_ads = atoms.cell
    pbc_ads = atoms.pbc

    ni_indices = [i for i, s in enumerate(symbols) if s == "Ni"]
    if ni_indices != list(range(n_ni)):
        raise ValueError(
            f"Frame {frame_idx}: Ni atoms are not the leading block of the frame; "
            "anchor indices would not map onto the bare reference."
        )
    o_idx, _h_oh_idx, h_ads_idx = find_adsorbate_indices(
        symbols, positions, cell_ads, pbc_ads
    )

    o_anchor_idx, o_site, o_anchor_meta = extract_o_anchor_indices_hybrid(
        ni_indices,
        positions,
        o_idx,
        cell_ads,
        pbc_ads,
    )
    h_anchor_idx, h_site, h_anchor_meta = extract_h_anchor_indices_hybrid(
        ni_indices,
        positions,
        h_ads_idx,
        cell_ads,
        pbc_ads,
    )

    if group_label not in initial_refs:
        raise ValueError(
            f"Frame {frame_idx}: group_label={group_label!r} not found in INITIAL_NI_FILES. "
            f"Available keys: {list(initial_refs.keys())}"
        )

    initial_atoms = initial_refs[group_label]
    initial_symbols, positions_initial = get_symbols_positions(initial_atoms)

    # Geometry, cell, and PBC all come from the same bare reference used to
    # construct the input graph.
    cell_initial = initial_atoms.cell
    pbc_initial = initial_atoms.pbc

    if domain_label == "SLAB":
        if not np.any(pbc_initial):
            raise ValueError(
                f"Frame {frame_idx}: periodic bare reference {group_label!r} "
                "has no PBC. Use an extxyz file containing Lattice and pbc metadata."
            )
        if not np.array_equal(
            np.asarray(pbc_initial, dtype=bool),
            np.asarray(pbc_ads, dtype=bool),
        ):
            raise ValueError(
                f"Frame {frame_idx}: PBC mismatch for {group_label!r}: "
                f"bare={list(map(bool, pbc_initial))}, "
                f"adsorbed={list(map(bool, pbc_ads))}."
            )
        if not np.allclose(
            np.asarray(cell_initial),
            np.asarray(cell_ads),
            rtol=0.0,
            atol=CELL_MATCH_ATOL,
        ):
            max_delta = float(
                np.max(np.abs(np.asarray(cell_initial) - np.asarray(cell_ads)))
            )
            raise ValueError(
                f"Frame {frame_idx}: cell mismatch for {group_label!r}; "
                f"maximum absolute difference is {max_delta:.3e} A "
                f"(tolerance {CELL_MATCH_ATOL:.1e} A)."
            )
    initial_ni_indices = [i for i, s in enumerate(initial_symbols) if s == "Ni"]

    if len(initial_ni_indices) != n_ni:
        raise ValueError(
            f"Frame {frame_idx}: bare reference {group_label} has {len(initial_ni_indices)} Ni atoms, "
            f"adsorbed frame has {n_ni}."
        )
    if len(o_anchor_idx) == 0:
        raise ValueError(f"Frame {frame_idx}: no O anchors found with NI_O_CUTOFF={NI_O_CUTOFF}")
    if len(h_anchor_idx) == 0:
        raise ValueError(f"Frame {frame_idx}: no H anchors found with NI_H_CUTOFF={NI_H_CUTOFF}")

    motif = o_site + h_site
    motif_id = MOTIF_TO_ID[motif]

    all_anchor_idx = sorted(set(o_anchor_idx) | set(h_anchor_idx))

    local_cutoff = get_local_cutoff(domain_label)
    edge_cutoff = get_edge_cutoff(domain_label)

    if domain_label == "SLAB":
        # Expand periodic images around the site using only the bare geometry.
        # The relaxed adsorbate frame supplies the anchor indices and target.
        o_shifts, h_shifts = resolve_anchor_shifts_from_adsorbates(
            o_anchor_idx,
            h_anchor_idx,
            positions,
            o_idx,
            h_ads_idx,
            positions_initial,
            cell_initial,
            pbc_initial,
        )
        images, x, local_positions, keep_ni = build_slab_local_graph(
            initial_ni_indices, positions_initial, o_anchor_idx, h_anchor_idx,
            positions_initial, cell_initial, pbc_initial, local_cutoff, edge_cutoff,
            o_shifts=o_shifts,
            h_shifts=h_shifts,
        )
        edge_index, edge_attr = build_edges_and_edge_attr(
            local_positions, cell_initial, [False, False, False], edge_cutoff,
        )
    else:
        keep_ni = select_bare_local_environment(
            initial_ni_indices,
            positions_initial,
            all_anchor_idx,
            cell_initial,
            pbc_initial,
            local_cutoff,
        )

        # Store the finite local patch in one unwrapped Cartesian image.
        local_positions = unwrap_positions_around_reference(
            positions_initial, keep_ni, cell_initial, pbc_initial, ref_idx=all_anchor_idx[0]
        )
        x = build_node_features(
            keep_ni, positions_initial, o_anchor_idx, h_anchor_idx,
            cell_initial, pbc_initial, edge_cutoff=edge_cutoff,
        )
        edge_index, edge_attr = build_edges_and_edge_attr(
            local_positions, cell_initial, pbc_initial, edge_cutoff,
        )


    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=torch.tensor(local_positions, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_energy=torch.tensor([e_ads], dtype=torch.float),
        y_motif=torch.tensor([motif_id], dtype=torch.long),
        sample_weight=torch.tensor([sample_weight], dtype=torch.float),
        frame_idx=torch.tensor([frame_idx], dtype=torch.long),
        n_ni=torch.tensor([n_ni], dtype=torch.long),
        num_nodes=torch.tensor([len(keep_ni)], dtype=torch.long),
    )

    meta = {
        "frame_idx": int(frame_idx),
        "domain": domain_label,
        "group_label": group_label,
        "config_label": str(atoms.info.get("config_label", "")),
        "n_ni": int(n_ni),
        "energy": float(e_ads),
        "motif": motif,
        "o_anchor_count": len(o_anchor_idx),
        "h_anchor_count": len(h_anchor_idx),
        "o_anchor_idx": list(map(int, o_anchor_idx)),
        "o_anchor_rule": o_anchor_meta["o_anchor_rule"],
        "o_short_contact_count": o_anchor_meta["o_short_contact_count"],
        "o_extended_contact_count": o_anchor_meta["o_extended_contact_count"],
        "o_short_contact_distances": o_anchor_meta["o_short_contact_distances"],
        "o_extended_contact_distances": o_anchor_meta["o_extended_contact_distances"],
        "h_anchor_idx": list(map(int, h_anchor_idx)),
        "h_anchor_rule": h_anchor_meta["h_anchor_rule"],
        "h_short_contact_count": h_anchor_meta["h_short_contact_count"],
        "h_extended_contact_count": h_anchor_meta["h_extended_contact_count"],
        "h_short_contact_distances": h_anchor_meta["h_short_contact_distances"],
        "h_extended_contact_distances": h_anchor_meta["h_extended_contact_distances"],
        "all_anchor_idx": list(map(int, all_anchor_idx)),
        "local_ni_count": len(keep_ni),
        "keep_ni": list(map(int, keep_ni)),
        "pbc_adsorbed": [bool(x) for x in pbc_ads],
        "pbc_bare": [bool(x) for x in pbc_initial],
        "sample_weight": float(sample_weight),
        "local_cutoff": float(local_cutoff),
        "edge_cutoff": float(edge_cutoff),
    }
    if domain_label == "SLAB":
        meta["keep_ni_shifts"] = [[int(b), [int(s) for s in shift]] for b, shift, _ in images]
    return data, meta

def build_pyg_dataset(extxyz_file: str, initial_refs, domain_label: str, sample_weight: float):
    """Build graphs and metadata for one nanoparticle or slab dataset."""
    loaded_frames = read(extxyz_file, index=":")
    print(f"Loaded {len(loaded_frames)} {domain_label} frames from {extxyz_file}")
    frames = filter_target_adsorption_state(loaded_frames, domain_label)

    available_keys = list(initial_refs.keys())
    group_labels = build_group_labels(frames, available_keys)

    dataset = []
    metadata = []
    motif_labels = []
    domain_labels = []

    for frame_idx, atoms in enumerate(frames):
        group_label = group_labels[frame_idx]
        data, meta = process_frame(
            atoms,
            initial_refs,
            group_label,
            frame_idx,
            domain_label=domain_label,
            sample_weight=sample_weight,
        )
        dataset.append(data)
        metadata.append(meta)
        motif_labels.append(meta["motif"])
        domain_labels.append(domain_label)

    return dataset, metadata, frames, group_labels, motif_labels, domain_labels

def split_one_domain(
    dataset,
    metadata,
    frames,
    group_labels,
    motif_labels,
    domain_labels,
    val_fraction,
    seed,
    fixed_indices=None,
):
    """Create a random split or apply exact indices from a fixed manifest."""
    combined = make_combined_strat_labels(domain_labels, group_labels, motif_labels)
    if fixed_indices is None:
        train_idx, val_idx = stratified_split(combined, val_fraction, seed=seed)
    else:
        train_idx = [int(i) for i in fixed_indices["train_idx"]]
        val_idx = [int(i) for i in fixed_indices["val_idx"]]
        expected = set(range(len(dataset)))
        train_set = set(train_idx)
        val_set = set(val_idx)
        if len(train_idx) != len(train_set) or len(val_idx) != len(val_set):
            raise ValueError("Fixed split contains duplicate indices.")
        if train_set & val_set:
            raise ValueError("Fixed train and validation indices overlap.")
        if train_set | val_set != expected:
            missing = sorted(expected - (train_set | val_set))
            extra = sorted((train_set | val_set) - expected)
            raise ValueError(
                "Fixed split does not cover the dataset exactly; "
                f"missing={missing[:10]}, extra={extra[:10]}."
            )
    return {
        "combined_labels": combined,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_dataset": [dataset[i] for i in train_idx],
        "val_dataset": [dataset[i] for i in val_idx],
        "train_frames": [frames[i] for i in train_idx],
        "val_frames": [frames[i] for i in val_idx],
        "train_metadata": [metadata[i] for i in train_idx],
        "val_metadata": [metadata[i] for i in val_idx],
    }


def load_fixed_split_manifest(path: str):
    """Load and minimally validate an exact NP/slab index manifest."""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    domains = manifest.get("domains")
    if not isinstance(domains, dict):
        raise ValueError(f"Fixed split manifest {path} has no 'domains' mapping.")
    for domain in ("NP", "SLAB"):
        if domain not in domains:
            raise ValueError(f"Fixed split manifest {path} is missing {domain}.")
        for key in ("train_idx", "val_idx"):
            if key not in domains[domain]:
                raise ValueError(f"Fixed split manifest {path} is missing {domain}.{key}.")
    print(f"Loaded fixed split from: {path}")
    return domains

# ============================================================
# MODEL
# ============================================================

class EdgeGatedConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="add")
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.res_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else None

    def forward(self, x, edge_index, edge_attr):
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_res = self.res_proj(x) if self.res_proj is not None else x
        out = self.node_mlp(torch.cat([x, m], dim=-1))
        return x_res + out

    def message(self, x_i, x_j, edge_attr):
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        a_ij = self.att_mlp(m_ij)
        return a_ij * m_ij

class BareSiteGCNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = EdgeGatedConv(node_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeGatedConv(hidden_dim, edge_dim, hidden_dim)
        self.conv3 = EdgeGatedConv(hidden_dim, edge_dim, hidden_dim)

        # Concatenate global, O-focused, and H-focused graph embeddings.
        pooled_dim = 3 * hidden_dim
        self.energy_head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def weighted_mean_pool(
        x: torch.Tensor,
        weights: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        weighted_sum = global_add_pool(x * weights, batch)
        weight_sum = global_add_pool(weights, batch)
        return weighted_sum / weight_sum.clamp_min(1e-8)

    def forward(self, x, edge_index, edge_attr, batch):
        # Preserve the input anchor flags and distances used for pooling.
        x_in = x
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = F.relu(self.conv3(h, edge_index, edge_attr))

        # Global pooling summarizes the complete local environment.
        g_global = global_mean_pool(h, batch)

        # O-focused pooling emphasizes O anchors and nearby Ni atoms.
        is_o_anchor = x_in[:, 1:2]
        dist_to_o = x_in[:, 4:5]
        weights_o = torch.exp(-dist_to_o / POOL_SIGMA) + ANCHOR_BOOST * is_o_anchor
        g_o = self.weighted_mean_pool(h, weights_o, batch)

        # H-focused pooling applies the same construction around H*.
        is_h_anchor = x_in[:, 2:3]
        dist_to_h = x_in[:, 5:6]
        weights_h = torch.exp(-dist_to_h / POOL_SIGMA) + ANCHOR_BOOST * is_h_anchor
        g_h = self.weighted_mean_pool(h, weights_h, batch)

        g = torch.cat([g_global, g_o, g_h], dim=-1)
        return self.energy_head(g).squeeze(-1)

# ============================================================
# TRAINING
# ============================================================

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.view(-1).to(pred.device)
    loss = (pred - target) ** 2
    return torch.sum(weights * loss) / torch.sum(weights).clamp_min(1e-8)

def weighted_mae(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.view(-1).to(pred.device)
    loss = torch.abs(pred - target)
    return torch.sum(weights * loss) / torch.sum(weights).clamp_min(1e-8)

def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_reg_loss = 0.0
    total_mae = 0.0
    total_weighted_mae = 0.0
    n_samples = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(DEVICE)
            e_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            e_true = batch.y_energy.view(-1)
            weights = (
                batch.sample_weight.view(-1)
                if hasattr(batch, "sample_weight")
                else torch.ones_like(e_true)
            )

            reg_loss = weighted_mse(e_pred, e_true, weights)
            loss = REG_LOSS_WEIGHT * reg_loss

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = e_true.size(0)
            total_loss += loss.item() * bs
            total_reg_loss += reg_loss.item() * bs
            total_mae += mae(e_pred, e_true).item() * bs
            total_weighted_mae += weighted_mae(e_pred, e_true, weights).item() * bs
            n_samples += bs

    return {
        "loss": total_loss / n_samples,
        "reg_loss": total_reg_loss / n_samples,
        "mae": total_mae / n_samples,
        "weighted_mae": total_weighted_mae / n_samples,
    }

def evaluate_per_motif(model, loader):
    model.eval()
    motif_errors = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            e_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            e_true = batch.y_energy.view(-1)
            m_true = batch.y_motif.view(-1)
            for ep, et, mt in zip(e_pred, e_true, m_true):
                motif = ID_TO_MOTIF[int(mt.item())]
                motif_errors[motif].append(abs(float(ep.item()) - float(et.item())))
    return {motif: {"count": len(vals), "mae_eV": float(np.mean(vals))} for motif, vals in sorted(motif_errors.items())}

def evaluate_by_domain(model, dataset: List[Data], metadata: List[dict], batch_size: int = BATCH_SIZE):
    if not dataset:
        return {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    rows = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            e_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            e_true = batch.y_energy.view(-1)
            for ep, et in zip(e_pred.cpu().numpy(), e_true.cpu().numpy()):
                meta = metadata[offset]
                rows.append({
                    "domain": meta["domain"],
                    "group_label": meta["group_label"],
                    "motif": meta["motif"],
                    "abs_error": abs(float(ep) - float(et)),
                    "signed_error": float(et) - float(ep),
                })
                offset += 1
    out = {}
    for key in ["domain", "group_label", "motif"]:
        grouped = defaultdict(list)
        for r in rows:
            grouped[r[key]].append(r["abs_error"])
        out[f"by_{key}"] = {
            str(k): {"count": len(v), "mae_eV": float(np.mean(v))}
            for k, v in sorted(grouped.items(), key=lambda kv: str(kv[0]))
        }
    return out

def concat(parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


# ============================================================
# FINE-TUNING HELPERS
# ============================================================

def load_pretrained_for_finetuning(
    model,
    pretrained_path: str,
    node_dim: int,
    edge_dim: int,
):
    """Load graph, pooling, and energy-head parameters for fine-tuning.

    The checkpoint must use the same node/edge features, hidden dimension,
    anchor definitions, and H2O-referenced target convention. Additional
    checkpoint keys are ignored, but every parameter required by the current
    energy model must be present with a matching shape.
    """
    ckpt = torch.load(pretrained_path, map_location=DEVICE)

    pretrained_settings = ckpt.get("settings", {})
    expected_anchor_settings = {
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
    }
    missing_anchor_settings = [
        key for key in expected_anchor_settings if key not in pretrained_settings
    ]
    if missing_anchor_settings:
        raise ValueError(
            "The pretrained checkpoint is not a compatible O+H hybrid checkpoint. "
            f"Missing settings: {missing_anchor_settings}"
        )

    for key, expected in expected_anchor_settings.items():
        actual = pretrained_settings[key]
        if float(actual) != float(expected):
            raise ValueError(
                f"Pretrained checkpoint anchor mismatch for {key}: "
                f"checkpoint has {actual}, current script expects {expected}."
            )

    checks = {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": HIDDEN_DIM,
    }
    for key, expected in checks.items():
        if key in ckpt and int(ckpt[key]) != int(expected):
            raise ValueError(
                f"Pretrained checkpoint mismatch for {key}: "
                f"checkpoint has {ckpt[key]}, current model expects {expected}."
            )

    pretrained_state = ckpt["model_state_dict"]
    new_state = model.state_dict()

    missing_or_mismatched = []
    compatible_state = {}
    for key, expected_tensor in new_state.items():
        if key not in pretrained_state:
            missing_or_mismatched.append(f"{key} (missing)")
            continue
        if pretrained_state[key].shape != expected_tensor.shape:
            missing_or_mismatched.append(
                f"{key} (checkpoint {tuple(pretrained_state[key].shape)} vs "
                f"current {tuple(expected_tensor.shape)})"
            )
            continue
        compatible_state[key] = pretrained_state[key]

    if missing_or_mismatched:
        raise RuntimeError(
            "The pretrained checkpoint cannot initialize the current energy-only model:\n  - "
            + "\n  - ".join(missing_or_mismatched)
        )

    model.load_state_dict(compatible_state, strict=True)

    print("\n=== FINE-TUNING ===")
    print(f"Loaded pretrained checkpoint: {pretrained_path}")
    if "best_val_mae_eV" in ckpt:
        print(f"Pretrained checkpoint validation MAE: {ckpt['best_val_mae_eV']} eV")
    print("Loaded all graph, pooling, and energy-head parameters.")

def maybe_freeze_message_passing(model):
    """Optionally freeze graph convolutions and train only the energy head."""
    if not FREEZE_MESSAGE_PASSING:
        return

    frozen_prefixes = ("conv1", "conv2", "conv3")
    n_frozen = 0
    n_trainable = 0

    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad = False
            n_frozen += param.numel()
        else:
            param.requires_grad = True
            n_trainable += param.numel()

    print("\n=== FREEZE MODE ===")
    print("Frozen layers: conv1, conv2, conv3")
    print(f"Frozen parameters:    {n_frozen}")
    print(f"Trainable parameters: {n_trainable}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Build the datasets, fine-tune the model, and save the best checkpoint."""
    initial_refs = load_initial_reference_structures()
    fixed_split = load_fixed_split_manifest(FIXED_SPLIT_JSON)

    domain_splits = {}
    all_dataset = []
    all_metadata = []
    all_frames = []
    all_group_labels = []
    all_motif_labels = []
    all_domain_labels = []

    # Build and split each domain independently before combining them.
    (
        np_dataset,
        np_metadata,
        np_frames,
        np_group_labels,
        np_motif_labels,
        np_domain_labels,
    ) = build_pyg_dataset(
        NP_EXTXYZ_FILE,
        initial_refs,
        domain_label="NP",
        sample_weight=NP_SAMPLE_WEIGHT,
    )
    domain_splits["NP"] = split_one_domain(
        np_dataset,
        np_metadata,
        np_frames,
        np_group_labels,
        np_motif_labels,
        np_domain_labels,
        VAL_FRACTION_NP,
        SEED,
        fixed_indices=None if fixed_split is None else fixed_split["NP"],
    )

    all_dataset.extend(np_dataset)
    all_metadata.extend(np_metadata)
    all_frames.extend(np_frames)
    all_group_labels.extend(np_group_labels)
    all_motif_labels.extend(np_motif_labels)
    all_domain_labels.extend(np_domain_labels)

    if SLAB_EXTXYZ_FILE is not None:
        (
            slab_dataset,
            slab_metadata,
            slab_frames,
            slab_group_labels,
            slab_motif_labels,
            slab_domain_labels,
        ) = build_pyg_dataset(
            SLAB_EXTXYZ_FILE,
            initial_refs,
            domain_label="SLAB",
            sample_weight=SLAB_SAMPLE_WEIGHT,
        )
        domain_splits["SLAB"] = split_one_domain(
            slab_dataset,
            slab_metadata,
            slab_frames,
            slab_group_labels,
            slab_motif_labels,
            slab_domain_labels,
            VAL_FRACTION_SLAB,
            SEED,
            fixed_indices=None if fixed_split is None else fixed_split["SLAB"],
        )

        all_dataset.extend(slab_dataset)
        all_metadata.extend(slab_metadata)
        all_frames.extend(slab_frames)
        all_group_labels.extend(slab_group_labels)
        all_motif_labels.extend(slab_motif_labels)
        all_domain_labels.extend(slab_domain_labels)
    else:
        slab_frames = []

    # Summarize the complete input dataset.
    domain_counts = Counter(all_domain_labels)
    group_counts = Counter(all_group_labels)
    motif_counts = Counter(all_motif_labels)
    o_counts = Counter(meta["o_anchor_count"] for meta in all_metadata)
    h_counts = Counter(meta["h_anchor_count"] for meta in all_metadata)
    combined_labels = make_combined_strat_labels(
        all_domain_labels, all_group_labels, all_motif_labels
    )
    combined_counts = Counter(combined_labels)

    print("\nDomain counts:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    print("\nGroup counts:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count}")
    print("\nMotif counts:")
    for motif, count in sorted(motif_counts.items()):
        print(f"  {motif}: {count}")

    print("\nO-anchor extraction rules:")
    o_rule_counts = Counter(meta["o_anchor_rule"] for meta in all_metadata)
    for rule, count in sorted(o_rule_counts.items()):
        print(f"  {rule}: {count}")

    print("\nH-anchor extraction rules:")
    h_rule_counts = Counter(meta["h_anchor_rule"] for meta in all_metadata)
    for rule, count in sorted(h_rule_counts.items()):
        print(f"  {rule}: {count}")

    train_dataset = concat(
        [domain_splits[d]["train_dataset"] for d in domain_splits]
    )
    val_dataset = concat([domain_splits[d]["val_dataset"] for d in domain_splits])
    train_frames = concat([domain_splits[d]["train_frames"] for d in domain_splits])
    val_frames = concat([domain_splits[d]["val_frames"] for d in domain_splits])
    train_metadata = concat(
        [domain_splits[d]["train_metadata"] for d in domain_splits]
    )
    val_metadata = concat(
        [domain_splits[d]["val_metadata"] for d in domain_splits]
    )

    # Persist the exact splits used for training and validation.
    write(TRAIN_EXTXYZ_OUT, train_frames, format="extxyz")
    write(VAL_EXTXYZ_OUT, val_frames, format="extxyz")
    if "NP" in domain_splits:
        write(TRAIN_NP_EXTXYZ_OUT, domain_splits["NP"]["train_frames"], format="extxyz")
        write(VAL_NP_EXTXYZ_OUT, domain_splits["NP"]["val_frames"], format="extxyz")
    if "SLAB" in domain_splits:
        write(TRAIN_SLAB_EXTXYZ_OUT, domain_splits["SLAB"]["train_frames"], format="extxyz")
        write(VAL_SLAB_EXTXYZ_OUT, domain_splits["SLAB"]["val_frames"], format="extxyz")

    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print("\nTrain by domain:")
    for domain, count in sorted(
        Counter(meta["domain"] for meta in train_metadata).items()
    ):
        print(f"  {domain}: {count}")
    print("\nVal by domain:")
    for domain, count in sorted(
        Counter(meta["domain"] for meta in val_metadata).items()
    ):
        print(f"  {domain}: {count}")

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    node_dim = all_dataset[0].x.shape[1]
    edge_dim = all_dataset[0].edge_attr.shape[1]

    model = BareSiteGCNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    if FINE_TUNE:
        load_pretrained_for_finetuning(
            model,
            PRETRAINED_MODEL_PATH,
            node_dim=node_dim,
            edge_dim=edge_dim,
        )

    maybe_freeze_message_passing(model)

    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {n_parameters:,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_score = float("inf")
    best_state = None
    patience_counter = 0

    print("\nStarting training...\n")
    print(f"Training mode: {'fine-tune' if FINE_TUNE else 'from scratch'}")
    print(f"LR = {LR}")
    print(f"EPOCHS = {EPOCHS}")
    print(f"EARLY_STOPPING_PATIENCE = {EARLY_STOPPING_PATIENCE}")
    print(f"NP_SAMPLE_WEIGHT = {NP_SAMPLE_WEIGHT}")
    print(f"SLAB_SAMPLE_WEIGHT = {SLAB_SAMPLE_WEIGHT}\n")
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, optimizer=None)

        # Use the unweighted validation MAE for model selection.
        val_score = val_metrics["mae"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train MAE: {train_metrics['mae']:.6f} eV | "
            f"Train wMAE: {train_metrics['weighted_mae']:.6f} eV | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val MAE: {val_metrics['mae']:.6f} eV | "
            f"Val wMAE: {val_metrics['weighted_mae']:.6f} eV"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "hidden_dim": HIDDEN_DIM,
                "n_parameters": int(n_parameters),
                "energy_only": True,
                "motif_to_id": MOTIF_TO_ID,
                "id_to_motif": ID_TO_MOTIF,
                "domain_splits": {
                    d: {"train_idx": domain_splits[d]["train_idx"], "val_idx": domain_splits[d]["val_idx"]}
                    for d in domain_splits
                },
                "group_labels": all_group_labels,
                "motif_labels": all_motif_labels,
                "domain_labels": all_domain_labels,
                "combined_labels": combined_labels,
                "settings": {
                    "NP_EXTXYZ_FILE": NP_EXTXYZ_FILE,
                    "SLAB_EXTXYZ_FILE": SLAB_EXTXYZ_FILE,
                    "FIXED_SPLIT_JSON": FIXED_SPLIT_JSON,
                    "FIXED_SPLIT": FIXED_SPLIT_JSON is not None,
                    "FIXED_SPLIT_CLOSEST_H_COMPATIBILITY": FIXED_SPLIT_JSON is not None,
                    "NP_SAMPLE_WEIGHT": NP_SAMPLE_WEIGHT,
                    "SLAB_SAMPLE_WEIGHT": SLAB_SAMPLE_WEIGHT,
                    "FINE_TUNE": FINE_TUNE,
                    "PRETRAINED_MODEL_PATH": PRETRAINED_MODEL_PATH,
                    "FREEZE_MESSAGE_PASSING": FREEZE_MESSAGE_PASSING,
                    "NI_O_CUTOFF": NI_O_CUTOFF,
                    "NI_O_FOLDED_CUTOFF": NI_O_FOLDED_CUTOFF,
                    "O_FOLDED_N_SHORT": O_FOLDED_N_SHORT,
                    "O_FOLDED_N_TOTAL": O_FOLDED_N_TOTAL,
                    "MAX_STANDARD_O_ANCHORS": MAX_STANDARD_O_ANCHORS,
                    "O_ANCHOR_DEFINITION": "hybrid_2p20_standard_at_least_2short_four_total_F",
                    "NI_H_CUTOFF": NI_H_CUTOFF,
                    "NI_H_FOLDED_CUTOFF": NI_H_FOLDED_CUTOFF,
                    "H_FOLDED_N_SHORT": H_FOLDED_N_SHORT,
                    "H_FOLDED_N_TOTAL": H_FOLDED_N_TOTAL,
                    "OH_BOND_CUTOFF": OH_BOND_CUTOFF,
                    "MAX_STANDARD_H_ANCHORS": MAX_STANDARD_H_ANCHORS,
                    "H_ANCHOR_DEFINITION": "hybrid_2p00_standard_at_least_2short_four_total_F",
                    "SITE_TYPES": list(SITE_TYPES),
                    "LOCAL_CUTOFF": LOCAL_CUTOFF,
                    "EDGE_CUTOFF": EDGE_CUTOFF,
                    "LOCAL_CUTOFF_BY_DOMAIN": {k: (None if v is None else float(v))
                                                for k, v in LOCAL_CUTOFF_BY_DOMAIN.items()},
                    "EDGE_CUTOFF_BY_DOMAIN":  {k: (None if v is None else float(v))
                                                for k, v in EDGE_CUTOFF_BY_DOMAIN.items()},
                    "N_RBF": N_RBF,
                    "RBF_DMIN": RBF_DMIN,
                    "RBF_DMAX": RBF_DMAX,
                    "RBF_GAMMA": RBF_GAMMA,
                    "BATCH_SIZE": BATCH_SIZE,
                    "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "SEED": SEED,
                    "VAL_FRACTION_NP": VAL_FRACTION_NP,
                    "VAL_FRACTION_SLAB": VAL_FRACTION_SLAB,
                    "REG_LOSS_WEIGHT": REG_LOSS_WEIGHT,
                    "TRAINING_OBJECTIVE": "energy_only",
                    "TARGET_REFERENCE": "H2O_gas",
                },
                "best_val_score": best_val_score,
                "best_val_mae_eV": val_metrics["mae"],
                "best_val_weighted_mae_eV": val_metrics["weighted_mae"],
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_per_motif": evaluate_per_motif(model, val_loader),
                "val_by_domain_group_motif": evaluate_by_domain(model, val_dataset, val_metadata),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training finished but no best model was saved.")

    torch.save(best_state, MODEL_OUT)

    stats = {
        "n_frames_total": len(all_frames),
        "n_frames_np": len(np_frames),
        "n_frames_slab": len(slab_frames),
        "domain_counts": summarize_counter(domain_counts),
        "group_counts": summarize_counter(group_counts),
        "motif_counts": summarize_counter(motif_counts),
        "o_anchor_counts": summarize_counter(o_counts),
        "o_anchor_rule_counts": summarize_counter(o_rule_counts),
        "h_anchor_counts": summarize_counter(h_counts),
        "h_anchor_rule_counts": summarize_counter(h_rule_counts),
        "combined_label_counts": summarize_counter(combined_counts),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "n_parameters": int(n_parameters),
        "training_objective": "energy_only",
        "target_reference": "H2O_gas",
        "fixed_split_json": FIXED_SPLIT_JSON,
        "fixed_split": FIXED_SPLIT_JSON is not None,
        "fixed_split_closest_h_compatibility": FIXED_SPLIT_JSON is not None,
        "train_domain_counts": summarize_counter(Counter(m["domain"] for m in train_metadata)),
        "val_domain_counts": summarize_counter(Counter(m["domain"] for m in val_metadata)),
        "best_val_mae_eV": float(best_state["best_val_mae_eV"]),
        "best_val_weighted_mae_eV": float(best_state["best_val_weighted_mae_eV"]),
        "val_per_motif": best_state["val_per_motif"],
        "val_by_domain_group_motif": best_state["val_by_domain_group_motif"],
        "example_metadata_first5": all_metadata[:5],
    }
    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved best model to: {MODEL_OUT}")
    print(f"Saved dataset statistics to: {STATS_OUT}")
    print(f"Best validation MAE: {best_state['best_val_mae_eV']:.6f} eV")
    print(f"Best validation weighted MAE: {best_state['best_val_weighted_mae_eV']:.6f} eV")

if __name__ == "__main__":
    main()
