#!/usr/bin/env python3
"""
Train ONE BINARY exact-retention GNN for nanoparticles + slabs using:

    shared GNN encoder
        ├── NP retention head
        └── SLAB retention head

NP and slab datasets are built and split independently, then their training
subsets are concatenated for a single joint training run. The graph encoder is
shared, while the final classifier head is selected from the known domain.
The domain label is NOT included in the graph/node features.

Initial structures:
    - by default, the initial motif and O/H anchor indices are read from the
      extxyz header;
    - alternatively, they can be reconstructed from geometry with the loose
      pre-relaxation cutoffs.

Final relaxed structures:
    - classified with the hybrid O/H T/B/H/F definition below.

Only valid OH*+H* pairs are retained. Both the initial and relaxed structures
must contain exactly one O-H bond, and unphysically short O-H contacts are
rejected before graph construction.

Cell and PBC information are read from the extxyz structures. Minimum-image
distances are used for site assignment, while slab graphs contain the periodic
images needed to represent the complete local environment around the anchors.

Binary training target:
    1 = retained_exact:
        same motif AND same O/H anchor sets after relaxation
    0 = migrated:
        the motif or either complete anchor set changed after relaxation

Default split:
    grouped independently within NP and SLAB by initial site.
"""

import copy
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.io import read
from ase.geometry import find_mic
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, MessagePassing

# ============================================================
# SETTINGS
# ============================================================

NP_INITIAL_EXTXYZ_FILE = "../datasets/dataset_NPs_retention_model_ini_stru.extxyz"
NP_FINAL_EXTXYZ_FILE = "../datasets/dataset_NPs_retention_model_relax_stru.extxyz"

SLAB_INITIAL_EXTXYZ_FILE = "../datasets/dataset_SLABs_retention_model_ini_stru.extxyz"
SLAB_FINAL_EXTXYZ_FILE = "../datasets/dataset_SLABs_retention_model_relax_stru.extxyz"

# ------------------------------------------------------------
# Initial candidate identity
# ------------------------------------------------------------
# True: read motif_label, o_anchor, and h_anchor from the initial extxyz header
# and treat these stored candidate descriptors as authoritative.
# False: determine the anchors from geometry with the loose cutoffs below and
# derive motif_label from the resulting anchor counts.
USE_HEADER_INITIAL_SITE = True

DOMAIN_FILES = {
    "NP": (NP_INITIAL_EXTXYZ_FILE, NP_FINAL_EXTXYZ_FILE),
    "SLAB": (SLAB_INITIAL_EXTXYZ_FILE, SLAB_FINAL_EXTXYZ_FILE),
}

DOMAIN_TO_ID = {"NP": 0, "SLAB": 1}
ID_TO_DOMAIN = {v: k for k, v in DOMAIN_TO_ID.items()}

# Relative contribution of each domain's BCE loss to the shared-encoder
# objective when both domains are present in a batch.
DOMAIN_LOSS_WEIGHT = {"NP": 0.80, "SLAB": 0.20}

MODEL_OUT = "retention_model.pt"
STATS_OUT = "retention_statistics.json"

# ------------------------------------------------------------
# Reproducible train-validation split
# ------------------------------------------------------------
# TRAINING_SEED controls weight initialization, DataLoader shuffling, dropout, etc.
TRAINING_SEED = 10

# SPLIT_SEED controls generation of a new grouped split.
SPLIT_SEED = 32

# For ablation or random-seed comparisons, set USE_FIXED_SPLIT=True so every run
# uses the same frame identities. If FIXED_SPLIT_FILE does not exist, the script
# creates it from SPLIT_SEED; subsequent runs load it.
# The checkpoint we provide was trained with a fixed split to compare its performance relative to the DFT-geometry case.
FIXED_SPLIT_FILE = "paper_splits/retention_checkpoint_split.json"
USE_FIXED_SPLIT = False

# Every run also saves a record of its active split here.
SPLIT_OUT = "split_used_this_run.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Initial anchor extraction
# ------------------------------------------------------------
# Pre-relaxation placements use broad cutoffs when geometric detection is enabled.
# All candidates are distance-sorted and at most 4 are retained, so the initial
# vocabulary remains T/B/H/F.
NI_O_CUTOFF_INITIAL = 2.45
NI_H_CUTOFF_INITIAL = 2.10
MAX_ANCHORS_PER_ADSORBATE = 4

# ------------------------------------------------------------
# Final relaxed hybrid motif definition
# ------------------------------------------------------------
NI_O_CUTOFF_FINAL = 2.20
NI_O_FOLDED_CUTOFF_FINAL = 2.30
O_FOLDED_N_SHORT_FINAL = 2
O_FOLDED_N_TOTAL_FINAL = 4
MAX_STANDARD_O_ANCHORS_FINAL = 3

NI_H_CUTOFF_FINAL = 2.00
NI_H_FOLDED_CUTOFF_FINAL = 2.20
H_FOLDED_N_SHORT_FINAL = 2
H_FOLDED_N_TOTAL_FINAL = 4
MAX_STANDARD_H_ANCHORS_FINAL = 3

# Exactly one O-H distance must be below OH_BOND_CUTOFF. The minimum-distance
# guard rejects overlapping atoms before the bond-count test is applied.
OH_BOND_CUTOFF = 1.20
MIN_OH_SEPARATION = 0.70

# ------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------
LOCAL_CUTOFF = 5.5
EDGE_CUTOFF = 4.0

# Same domain-specific overrides used in the NP+slab adsorption-energy workflow.
LOCAL_CUTOFF_BY_DOMAIN = {
    "NP": None,
    "SLAB": 6.0,
}
EDGE_CUTOFF_BY_DOMAIN = {
    "NP": None,
    "SLAB": 4.5,
}


def get_local_cutoff(domain_label: str) -> float:
    value = LOCAL_CUTOFF_BY_DOMAIN.get(domain_label)
    return float(value) if value is not None else LOCAL_CUTOFF


def get_edge_cutoff(domain_label: str) -> float:
    value = EDGE_CUTOFF_BY_DOMAIN.get(domain_label)
    return float(value) if value is not None else EDGE_CUTOFF


# Distance basis expansion
N_RBF = 16
RBF_DMIN = 0.0
RBF_DMAX = 5.0
RBF_GAMMA = 10.0

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
BATCH_SIZE = 32
LR = 8e-4
WEIGHT_DECAY = 1e-6
EPOCHS = 200
VAL_FRACTION_NP = 0.20
VAL_FRACTION_SLAB = 0.20
EARLY_STOPPING_PATIENCE = 25
HIDDEN_DIM = 64

# NP and slab are split independently with the same grouping rule.
SPLIT_MODE = "initial_site"

# "loss" or "balanced_accuracy".
# For this binary probability model, loss is the recommended default because
# ordinary (unweighted) BCE is a proper probabilistic scoring rule.
# NP and SLAB validation values are averaged so that one domain cannot hide poor
# performance in the other simply by containing more data.
EARLY_STOPPING_METRIC = "loss"

# The binary objective uses unweighted BCE so the output can be interpreted as a
# retained-site probability at the threshold below.
BINARY_THRESHOLD = 0.50

# Pooling
POOL_SIGMA = 1.0
ANCHOR_BOOST = 1.0

# ============================================================
# MOTIF AND BINARY-LABEL DEFINITIONS
# ============================================================

MOTIF_TO_ID = {
    "TT": 0,
    "TB": 1,
    "TH": 2,
    "TF": 3,
    "BT": 4,
    "BB": 5,
    "BH": 6,
    "BF": 7,
    "HT": 8,
    "HB": 9,
    "HH": 10,
    "HF": 11,
    "FT": 12,
    "FB": 13,
    "FH": 14,
    "FF": 15,
}
ID_TO_MOTIF = {v: k for k, v in MOTIF_TO_ID.items()}

STABLE = 1
UNSTABLE = 0

# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(TRAINING_SEED)

# ============================================================
# PBC HELPERS
# ============================================================


def mic_vectors(vectors: np.ndarray, cell, pbc) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    if not np.any(pbc):
        return vectors
    vectors_mic, _ = find_mic(vectors, cell=cell, pbc=pbc)
    return np.asarray(vectors_mic)


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


def center_of_indices_pbc(
    positions: np.ndarray,
    indices: List[int],
    cell,
    pbc,
) -> np.ndarray:
    return unwrap_positions_around_reference(positions, indices, cell, pbc).mean(axis=0)


# ============================================================
# GEOMETRY HELPERS
# ============================================================


def get_symbols_positions(atoms) -> Tuple[List[str], np.ndarray]:
    return atoms.get_chemical_symbols(), atoms.get_positions()


def gaussian_distance_expansion(
    distances: np.ndarray,
    dmin: float = RBF_DMIN,
    dmax: float = RBF_DMAX,
    n_kernels: int = N_RBF,
    gamma: float = RBF_GAMMA,
) -> np.ndarray:
    centers = np.linspace(dmin, dmax, n_kernels)
    diff = distances[:, None] - centers[None, :]
    return np.exp(-gamma * diff**2)


def classify_site(n_neighbors: int) -> str:
    if n_neighbors <= 1:
        return "T"
    if n_neighbors == 2:
        return "B"
    if n_neighbors == 3:
        return "H"
    return "F"


def find_adsorbate_indices(
    symbols: List[str],
    positions: np.ndarray,
    cell,
    pbc,
) -> Tuple[int, int, int]:
    """Return O, hydroxyl-H, and H* indices for a valid OH*+H* structure.

    Distances use the minimum-image convention for periodic structures. Frames
    are rejected when OH* is dissociated, both H atoms form an H2O-like species,
    or any O-H separation is unphysically short.
    """
    o_indices = [i for i, sym in enumerate(symbols) if sym == "O"]
    h_indices = [i for i, sym in enumerate(symbols) if sym == "H"]

    if len(o_indices) != 1:
        raise ValueError(f"Expected 1 O, found {len(o_indices)}")
    if len(h_indices) != 2:
        raise ValueError(f"Expected 2 H, found {len(h_indices)}")

    o_idx = o_indices[0]
    o_pos = positions[o_idx]

    distances = sorted(
        [
            (h_idx, distance_pbc(positions[h_idx], o_pos, cell, pbc))
            for h_idx in h_indices
        ],
        key=lambda item: item[1],
    )
    distance_values = [distance for _, distance in distances]

    if distance_values[0] < MIN_OH_SEPARATION:
        raise ValueError(
            f"Unphysically short O-H contact below {MIN_OH_SEPARATION:.2f} Å; "
            f"distances={distance_values}"
        )

    oh_like = [
        (h_idx, distance) for h_idx, distance in distances if distance < OH_BOND_CUTOFF
    ]
    if len(oh_like) == 0:
        raise ValueError(
            f"Dissociated O*+H*+H*: no O-H distance below "
            f"{OH_BOND_CUTOFF:.2f} Å; distances={distance_values}"
        )
    if len(oh_like) == 2:
        raise ValueError(
            f"H2O-like structure: two O-H distances below "
            f"{OH_BOND_CUTOFF:.2f} Å; distances={distance_values}"
        )

    h_oh_idx = oh_like[0][0]
    h_ads_idx = next(h_idx for h_idx in h_indices if h_idx != h_oh_idx)

    return o_idx, h_oh_idx, h_ads_idx


def extract_initial_anchor_indices(
    ni_indices: List[int],
    positions: np.ndarray,
    ads_idx: int,
    cutoff: float,
    cell,
    pbc,
    max_anchors: int = MAX_ANCHORS_PER_ADSORBATE,
) -> List[int]:
    """Return the closest Ni anchors inside a loose pre-relaxation cutoff."""
    ads_pos = positions[ads_idx]
    hits = []
    for ni_idx in ni_indices:
        d = distance_pbc(positions[ni_idx], ads_pos, cell, pbc)
        if d < cutoff:
            hits.append((ni_idx, d))
    hits.sort(key=lambda item: item[1])

    return [idx for idx, _ in hits[:max_anchors]]


def read_initial_site_from_header(atoms) -> Tuple[str, List[int], List[int]]:
    """Read and validate the initial motif and zero-based Ni anchor indices."""
    required = ("motif_label", "o_anchor", "h_anchor")
    missing = [key for key in required if key not in atoms.info]
    if missing:
        raise ValueError(
            "Initial structure is missing required header field(s): "
            + ", ".join(missing)
        )

    try:
        o_anchors = sorted(int(v) for v in np.atleast_1d(atoms.info["o_anchor"]))
        h_anchors = sorted(int(v) for v in np.atleast_1d(atoms.info["h_anchor"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("Could not parse o_anchor/h_anchor from the header.") from exc

    motif = str(atoms.info["motif_label"]).upper()
    if motif not in MOTIF_TO_ID:
        raise ValueError(f"Unsupported initial motif_label={motif!r}.")
    if not o_anchors or not h_anchors:
        raise ValueError("Initial o_anchor and h_anchor must both be non-empty.")

    symbols = atoms.get_chemical_symbols()
    for adsorbate, anchors in (("O", o_anchors), ("H", h_anchors)):
        invalid = [
            idx
            for idx in anchors
            if idx < 0 or idx >= len(symbols) or symbols[idx] != "Ni"
        ]
        if invalid:
            raise ValueError(
                f"Initial {adsorbate} anchor indices are not valid Ni atoms: {invalid}"
            )

    motif_from_anchors = classify_site(len(o_anchors)) + classify_site(len(h_anchors))
    if motif != motif_from_anchors:
        raise ValueError(
            f"Header motif_label={motif} disagrees with the stored anchor sets "
            f"({motif_from_anchors})."
        )

    return motif, o_anchors, h_anchors


def select_bare_local_environment(
    ni_indices: List[int],
    positions: np.ndarray,
    anchor_indices: List[int],
    cell,
    pbc,
    cutoff: float = LOCAL_CUTOFF,
) -> List[int]:
    if not anchor_indices:
        raise ValueError("No anchors found.")

    keep = set(anchor_indices)
    anchor_positions = positions[anchor_indices]

    for ni_idx in ni_indices:
        ni_pos = positions[ni_idx]
        dmin = float(
            np.min(mic_distances(anchor_positions - ni_pos[None, :], cell, pbc))
        )
        if dmin <= cutoff:
            keep.add(ni_idx)

    return sorted(keep)


def compute_coordination_numbers(
    ni_positions: np.ndarray,
    cell,
    pbc,
    cutoff: float = EDGE_CUTOFF,
) -> np.ndarray:
    """Coordination numbers inside the selected local Ni subgraph, PBC-aware."""
    n = len(ni_positions)
    cn = np.zeros(n, dtype=np.float32)

    for i in range(n):
        count = 0
        for j in range(n):
            if i == j:
                continue
            if distance_pbc(ni_positions[j], ni_positions[i], cell, pbc) <= cutoff:
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
    local_positions = positions[keep_indices]
    cn = compute_coordination_numbers(
        local_positions, cell=cell, pbc=pbc, cutoff=edge_cutoff
    )

    o_set = set(o_anchor_idx)
    h_set = set(h_anchor_idx)
    any_set = o_set | h_set

    o_center = center_of_indices_pbc(positions, o_anchor_idx, cell, pbc)
    h_center = center_of_indices_pbc(positions, h_anchor_idx, cell, pbc)
    any_center = center_of_indices_pbc(positions, sorted(any_set), cell, pbc)

    feats = []
    for local_i, global_idx in enumerate(keep_indices):
        pos = positions[global_idx]
        feats.append(
            [
                1.0,
                1.0 if global_idx in o_set else 0.0,
                1.0 if global_idx in h_set else 0.0,
                1.0 if global_idx in any_set else 0.0,
                distance_pbc(pos, o_center, cell, pbc),
                distance_pbc(pos, h_center, cell, pbc),
                distance_pbc(pos, any_center, cell, pbc),
                float(cn[local_i]),
            ]
        )

    return np.asarray(feats, dtype=np.float32)


def build_edges_and_edge_attr(
    local_positions: np.ndarray,
    cell,
    pbc,
    cutoff: float = EDGE_CUTOFF,
):
    edge_src, edge_dst, distances = [], [], []
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

    if not distances:
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
#
# Slab anchors are first placed in one consistent periodic image. The local
# environment is then expanded into the required neighbouring cell images and
# represented as an ordinary finite Euclidean graph. Nodes are identified by
# (base_Ni_index, cell_shift), so distinct images of the same atom can both enter
# the local graph when required by the cutoff.


def _mst_unwrap_group(
    indices: List[int],
    positions_ads: np.ndarray,
    cell,
    pbc,
) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[int, np.ndarray]]:
    """Place one adsorbate's anchor group around a common reference image."""
    if not indices:
        return {}, {}
    cell_arr = np.asarray(cell)
    inv_cell = np.linalg.inv(cell_arr)

    ref = indices[0]
    ref_pos = positions_ads[ref]
    canonical_pos: Dict[int, np.ndarray] = {ref: ref_pos}
    shifts: Dict[int, Tuple[int, int, int]] = {ref: (0, 0, 0)}

    for idx in indices[1:]:
        if idx in shifts:
            continue
        raw_vec = positions_ads[idx] - ref_pos
        mic_vec = mic_vectors(raw_vec[None, :], cell, pbc)[0]
        canonical_pos[idx] = ref_pos + mic_vec
        shift_cart = mic_vec - raw_vec
        frac = shift_cart @ inv_cell
        shifts[idx] = tuple(int(round(v)) for v in frac)
    return shifts, canonical_pos


def resolve_anchor_shifts(
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    positions_ads: np.ndarray,
    cell,
    pbc,
) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]]:
    """Resolve O- and H-anchor images while preserving each site's geometry.

    The two anchor groups are unwrapped independently and the H group is then
    translated as a unit to its nearest periodic position relative to the O
    group. Separate shift maps preserve both roles of a Ni atom shared by the
    O and H sites.
    """
    cell_arr = np.asarray(cell)
    inv_cell = np.linalg.inv(cell_arr)

    o_shifts, o_canonical = _mst_unwrap_group(o_anchor_idx, positions_ads, cell, pbc)
    h_shifts, h_canonical = _mst_unwrap_group(h_anchor_idx, positions_ads, cell, pbc)

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


def build_slab_local_graph(
    ni_indices: List[int],
    positions_bare: np.ndarray,
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    positions_ads: np.ndarray,
    cell,
    pbc,
    local_cutoff: float,
    edge_cutoff: float,
) -> Tuple[
    List[Tuple[int, Tuple[int, int, int], np.ndarray]],
    np.ndarray,
    np.ndarray,
    List[int],
]:
    """Explicit periodic-image local environment for one slab adsorption
    site. Returns (images, x, local_positions, keep_ni_debug):
      images          -- list of (base_ni_index, integer_shift, cart_pos)
      x               -- (N, 8) node feature array, same 8 columns as
                          build_node_features (bias, is_o_anchor,
                          is_h_anchor, is_any_anchor, dist_to_o_center,
                          dist_to_h_center, dist_to_any_center, CN)
      local_positions -- (N, 3) Cartesian positions, one row per image
      keep_ni_debug   -- base indices only, one per image (may repeat a
                         base index if more than one of its periodic copies
                         fell inside the crop) -- for logging/meta only
    """
    o_shifts, h_shifts = resolve_anchor_shifts(
        o_anchor_idx, h_anchor_idx, positions_ads, cell, pbc
    )

    cell_arr = np.asarray(cell)

    def _cart_of(b: int, shift: Tuple[int, int, int]) -> np.ndarray:
        return positions_bare[b] + np.array(shift) @ cell_arr

    # Keep the cell shift in each anchor key so shared base atoms retain the
    # correct role in both adsorption sites.
    o_anchor_images = {(b, o_shifts[b]) for b in o_anchor_idx}
    h_anchor_images = {(b, h_shifts[b]) for b in h_anchor_idx}
    any_anchor_images = o_anchor_images | h_anchor_images

    anchor_cart = np.array([_cart_of(b, s) for (b, s) in any_anchor_images])
    o_center = np.mean([_cart_of(b, s) for (b, s) in o_anchor_images], axis=0)
    h_center = np.mean([_cart_of(b, s) for (b, s) in h_anchor_images], axis=0)
    any_center = np.mean([_cart_of(b, s) for (b, s) in any_anchor_images], axis=0)

    pbc_arr = np.asarray(pbc, dtype=bool)
    reps = [0, 0, 0]
    for d in range(3):
        if pbc_arr[d]:
            reps[d] = int(np.ceil(local_cutoff / np.linalg.norm(cell_arr[d]))) + 1
    ranges = [range(-r, r + 1) if r else range(1) for r in reps]

    images: List[Tuple[int, Tuple[int, int, int], np.ndarray]] = []
    for b in ni_indices:
        base_pos = positions_bare[b]
        for sx in ranges[0]:
            for sy in ranges[1]:
                for sz in ranges[2]:
                    shift = (sx, sy, sz)
                    cart = base_pos + np.array(shift) @ cell_arr
                    dmin = float(
                        np.min(np.linalg.norm(anchor_cart - cart[None, :], axis=1))
                    )
                    if dmin <= local_cutoff:
                        images.append((b, shift, cart))

    if not images:
        raise ValueError(
            "No Ni images found within local_cutoff of the anchor cluster."
        )

    local_positions = np.array([im[2] for im in images], dtype=float)

    # Once the image cloud is built, graph distances and CN use ordinary
    # Euclidean distances within this finite local environment.
    cn_local = compute_coordination_numbers(
        local_positions,
        cell=cell_arr,
        pbc=[False, False, False],
        cutoff=edge_cutoff,
    )

    feats = []
    for (b, shift, cart), cn_i in zip(images, cn_local):
        key = (b, shift)
        feats.append(
            [
                1.0,
                1.0 if key in o_anchor_images else 0.0,
                1.0 if key in h_anchor_images else 0.0,
                1.0 if key in any_anchor_images else 0.0,
                float(np.linalg.norm(cart - o_center)),
                float(np.linalg.norm(cart - h_center)),
                float(np.linalg.norm(cart - any_center)),
                float(cn_i),
            ]
        )
    x = np.asarray(feats, dtype=np.float32)

    keep_ni_debug = [im[0] for im in images]
    return images, x, local_positions, keep_ni_debug


# ============================================================
# MOTIF EXTRACTION AND DATASET BUILDING
# ============================================================


def extract_anchor_indices_hybrid(
    ni_indices: List[int],
    positions: np.ndarray,
    ads_idx: int,
    short_cutoff: float,
    extended_cutoff: float,
    folded_n_short: int,
    folded_n_total: int,
    max_standard_anchors: int,
    cell,
    pbc,
) -> Tuple[List[int], str, Dict[str, object]]:
    """Extract final T/B/H/F anchors with the hybrid cutoff definition."""
    ads_pos = positions[ads_idx]
    extended_hits = []

    for ni_idx in ni_indices:
        d = distance_pbc(positions[ni_idx], ads_pos, cell, pbc)
        if d < extended_cutoff:
            extended_hits.append((ni_idx, d))

    extended_hits.sort(key=lambda item: item[1])
    short_hits = [(ni_idx, d) for ni_idx, d in extended_hits if d < short_cutoff]

    is_folded = (
        len(short_hits) >= folded_n_short and len(extended_hits) == folded_n_total
    )

    if is_folded:
        selected_hits = extended_hits[:folded_n_total]
        site = "F"
        rule = "hybrid_folded"
    else:
        selected_hits = short_hits[:max_standard_anchors]
        site = classify_site(len(selected_hits))
        rule = "standard_short_cutoff"

    metadata = {
        "rule": rule,
        "short_contact_count": int(len(short_hits)),
        "extended_contact_count": int(len(extended_hits)),
        "short_contact_distances": [float(d) for _, d in short_hits],
        "extended_contact_distances": [float(d) for _, d in extended_hits],
    }
    return [idx for idx, _ in selected_hits], site, metadata


def get_final_motif_and_anchors_hybrid(
    atoms,
) -> Tuple[str, List[int], List[int], dict]:
    """Classify the relaxed structure with the final hybrid O/H rules."""
    symbols, positions = get_symbols_positions(atoms)
    cell = atoms.cell
    pbc = atoms.pbc
    ni_indices = [i for i, sym in enumerate(symbols) if sym == "Ni"]
    o_idx, _, h_ads_idx = find_adsorbate_indices(symbols, positions, cell, pbc)

    o_anchors, o_site, o_meta = extract_anchor_indices_hybrid(
        ni_indices=ni_indices,
        positions=positions,
        ads_idx=o_idx,
        short_cutoff=NI_O_CUTOFF_FINAL,
        extended_cutoff=NI_O_FOLDED_CUTOFF_FINAL,
        folded_n_short=O_FOLDED_N_SHORT_FINAL,
        folded_n_total=O_FOLDED_N_TOTAL_FINAL,
        max_standard_anchors=MAX_STANDARD_O_ANCHORS_FINAL,
        cell=cell,
        pbc=pbc,
    )
    h_anchors, h_site, h_meta = extract_anchor_indices_hybrid(
        ni_indices=ni_indices,
        positions=positions,
        ads_idx=h_ads_idx,
        short_cutoff=NI_H_CUTOFF_FINAL,
        extended_cutoff=NI_H_FOLDED_CUTOFF_FINAL,
        folded_n_short=H_FOLDED_N_SHORT_FINAL,
        folded_n_total=H_FOLDED_N_TOTAL_FINAL,
        max_standard_anchors=MAX_STANDARD_H_ANCHORS_FINAL,
        cell=cell,
        pbc=pbc,
    )

    if not o_anchors:
        raise ValueError(
            f"No final O anchors with short cutoff {NI_O_CUTOFF_FINAL:.2f} Å"
        )
    if not h_anchors:
        raise ValueError(
            f"No final H anchors with short cutoff {NI_H_CUTOFF_FINAL:.2f} Å"
        )

    motif = o_site + h_site
    return motif, o_anchors, h_anchors, {"o": o_meta, "h": h_meta}


def get_base_group_label(atoms) -> str:
    """Extract the structure family from the required config_label field."""
    import re as _re

    config_label = str(atoms.info.get("config_label", ""))
    if not config_label:
        raise ValueError("Each structure must define config_label in its header.")

    match = _re.search(r"(Ni\d+_conf\d+)", config_label)
    if match:
        return match.group(1)

    # Generic fallback for labels ending in a sequential frame index.
    return _re.sub(r"_\d+$", "", config_label)


def get_group_label(atoms, domain_label: str) -> str:
    """
    NP: use the nanoparticle/configuration family from config_label.

    SLAB: append the Ni atom count so e.g. Ni111 2x2 and Ni111 3x3 do not become
    the same split group when the header only says config_label=Ni111_....
    """
    base = get_base_group_label(atoms)
    if domain_label == "SLAB":
        n_ni = sum(sym == "Ni" for sym in atoms.get_chemical_symbols())
        return f"{base}_N{n_ni}"
    return base


def build_group_labels(frames, domain_label: str) -> List[str]:
    return [get_group_label(atoms, domain_label) for atoms in frames]


def anchors_key(anchor_list: List[int]) -> str:
    return "-".join(map(str, sorted(map(int, anchor_list))))


def process_frame_pair(
    initial_atoms,
    final_atoms,
    group_label: str,
    frame_idx: int,
    domain_label: str,
) -> Tuple[Data, dict]:
    """
    Graph input: local Ni subgraph built from the initial structure.
    Initial anchors + motif: extxyz header or loose geometry cutoffs.
    Final anchors + motif: hybrid relaxed-geometry definition.
    """
    init_sym, init_pos = get_symbols_positions(initial_atoms)
    init_cell = initial_atoms.cell
    init_pbc = initial_atoms.pbc
    ni_indices = [i for i, sym in enumerate(init_sym) if sym == "Ni"]
    n_ni = len(ni_indices)

    final_sym, final_pos = get_symbols_positions(final_atoms)
    final_n_ni = sum(sym == "Ni" for sym in final_sym)
    if final_n_ni != n_ni:
        raise ValueError(
            f"Frame {frame_idx}: initial has {n_ni} Ni atoms but final has {final_n_ni}."
        )

    config_label = str(initial_atoms.info.get("config_label", ""))
    final_config_label = str(final_atoms.info.get("config_label", ""))
    if not config_label or not final_config_label:
        raise ValueError(f"Frame {frame_idx}: config_label is required in both files.")
    if config_label != final_config_label:
        raise ValueError(
            f"Frame {frame_idx}: initial/final config_label mismatch "
            f"({config_label!r} != {final_config_label!r})."
        )

    # Validate the initial adsorbate state even when candidate anchors are read
    # from the header. The relaxed state is validated by the same function in
    # get_final_motif_and_anchors_hybrid().
    o_idx, _, h_ads_idx = find_adsorbate_indices(
        init_sym, init_pos, init_cell, init_pbc
    )
    find_adsorbate_indices(
        final_sym,
        final_pos,
        final_atoms.cell,
        final_atoms.pbc,
    )

    if USE_HEADER_INITIAL_SITE:
        motif_initial, o_anchor_idx, h_anchor_idx = read_initial_site_from_header(
            initial_atoms
        )
        initial_site_source = "header"
    else:
        o_anchor_idx = extract_initial_anchor_indices(
            ni_indices,
            init_pos,
            o_idx,
            NI_O_CUTOFF_INITIAL,
            init_cell,
            init_pbc,
        )
        h_anchor_idx = extract_initial_anchor_indices(
            ni_indices,
            init_pos,
            h_ads_idx,
            NI_H_CUTOFF_INITIAL,
            init_cell,
            init_pbc,
        )
        motif_initial = classify_site(len(o_anchor_idx)) + classify_site(
            len(h_anchor_idx)
        )
        initial_site_source = "geometry_loose_cutoffs"

    if not o_anchor_idx:
        raise ValueError(f"Frame {frame_idx}: no initial O anchors found.")
    if not h_anchor_idx:
        raise ValueError(f"Frame {frame_idx}: no initial H anchors found.")

    all_anchor_idx = sorted(set(o_anchor_idx) | set(h_anchor_idx))

    local_cutoff = get_local_cutoff(domain_label)
    edge_cutoff = get_edge_cutoff(domain_label)

    if domain_label == "SLAB":
        # Build a finite local graph containing the required periodic images.
        images, x, local_positions, keep_ni = build_slab_local_graph(
            ni_indices,
            init_pos,
            o_anchor_idx,
            h_anchor_idx,
            init_pos,
            init_cell,
            init_pbc,
            local_cutoff,
            edge_cutoff,
        )
        edge_index, edge_attr = build_edges_and_edge_attr(
            local_positions,
            init_cell,
            [False, False, False],
            edge_cutoff,
        )
    else:
        keep_ni = select_bare_local_environment(
            ni_indices,
            init_pos,
            all_anchor_idx,
            init_cell,
            init_pbc,
            local_cutoff,
        )

        # Unwrapped positions are stored for debugging/visualization; feature and edge
        # distances remain MIC/PBC-aware.
        local_positions = unwrap_positions_around_reference(
            init_pos,
            keep_ni,
            init_cell,
            init_pbc,
            ref_idx=all_anchor_idx[0],
        )

        x = build_node_features(
            keep_ni,
            init_pos,
            o_anchor_idx,
            h_anchor_idx,
            init_cell,
            init_pbc,
            edge_cutoff=edge_cutoff,
        )
        edge_index, edge_attr = build_edges_and_edge_attr(
            local_positions,
            init_cell,
            init_pbc,
            edge_cutoff,
        )

    motif_final, o_anchor_final, h_anchor_final, final_anchor_meta = (
        get_final_motif_and_anchors_hybrid(final_atoms)
    )

    same_motif = motif_initial == motif_final
    same_o_anchor = set(o_anchor_idx) == set(o_anchor_final)
    same_h_anchor = set(h_anchor_idx) == set(h_anchor_final)
    y_stable = STABLE if (same_motif and same_o_anchor and same_h_anchor) else UNSTABLE

    domain_id = DOMAIN_TO_ID[domain_label]

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=torch.tensor(local_positions, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_stable=torch.tensor([y_stable], dtype=torch.float),
        y_motif_initial=torch.tensor([MOTIF_TO_ID[motif_initial]], dtype=torch.long),
        y_motif_final=torch.tensor([MOTIF_TO_ID[motif_final]], dtype=torch.long),
        domain_id=torch.tensor([domain_id], dtype=torch.long),
        frame_idx=torch.tensor([frame_idx], dtype=torch.long),
        n_ni=torch.tensor([n_ni], dtype=torch.long),
        n_local_nodes=torch.tensor([len(keep_ni)], dtype=torch.long),
    )

    meta = {
        "frame_idx": int(frame_idx),
        "domain": domain_label,
        "domain_id": int(domain_id),
        "n_ni": int(n_ni),
        "group": group_label,
        "config_label": config_label,
        "motif_initial": motif_initial,
        "initial_site_source": initial_site_source,
        "motif_final": motif_final,
        "y_stable": int(y_stable),
        "same_motif": bool(same_motif),
        "same_o_anchor": bool(same_o_anchor),
        "same_h_anchor": bool(same_h_anchor),
        "o_anchor_initial": list(map(int, o_anchor_idx)),
        "h_anchor_initial": list(map(int, h_anchor_idx)),
        "o_anchor_final": list(map(int, o_anchor_final)),
        "h_anchor_final": list(map(int, h_anchor_final)),
        "o_anchor_final_rule": final_anchor_meta["o"]["rule"],
        "h_anchor_final_rule": final_anchor_meta["h"]["rule"],
        "o_short_contact_count_final": final_anchor_meta["o"]["short_contact_count"],
        "o_extended_contact_count_final": final_anchor_meta["o"][
            "extended_contact_count"
        ],
        "h_short_contact_count_final": final_anchor_meta["h"]["short_contact_count"],
        "h_extended_contact_count_final": final_anchor_meta["h"][
            "extended_contact_count"
        ],
        "o_short_contact_distances_final": final_anchor_meta["o"][
            "short_contact_distances"
        ],
        "o_extended_contact_distances_final": final_anchor_meta["o"][
            "extended_contact_distances"
        ],
        "h_short_contact_distances_final": final_anchor_meta["h"][
            "short_contact_distances"
        ],
        "h_extended_contact_distances_final": final_anchor_meta["h"][
            "extended_contact_distances"
        ],
        "o_anchor_count": len(o_anchor_idx),
        "h_anchor_count": len(h_anchor_idx),
        "pbc_initial": [bool(v) for v in init_pbc],
        "pbc_final": [bool(v) for v in final_atoms.pbc],
        "local_cutoff": float(local_cutoff),
        "edge_cutoff": float(edge_cutoff),
    }

    meta["initial_site_key"] = (
        f"{group_label}__init_{motif_initial}__"
        f"O_{anchors_key(o_anchor_idx)}__H_{anchors_key(h_anchor_idx)}"
    )
    meta["final_basin_key"] = (
        f"{group_label}__final_{motif_final}__"
        f"O_{anchors_key(o_anchor_final)}__H_{anchors_key(h_anchor_final)}"
    )
    meta["initial_to_final_key"] = (
        meta["initial_site_key"] + "__TO__" + meta["final_basin_key"]
    )

    return data, meta


def build_dataset(
    initial_frames,
    final_frames,
    group_labels,
    domain_label: str,
):
    if len(initial_frames) != len(final_frames):
        raise ValueError(
            f"{domain_label}: initial ({len(initial_frames)}) and final "
            f"({len(final_frames)}) extxyz files have different lengths."
        )

    dataset, metadata = [], []
    skipped = 0

    for frame_idx, (init_atoms, final_atoms) in enumerate(
        zip(initial_frames, final_frames)
    ):
        try:
            data, meta = process_frame_pair(
                init_atoms,
                final_atoms,
                group_labels[frame_idx],
                frame_idx,
                domain_label,
            )
            dataset.append(data)
            metadata.append(meta)
        except Exception as exc:
            print(f"  WARNING [{domain_label}] frame {frame_idx} skipped — {exc}")
            skipped += 1

    print(f"Built {len(dataset)} {domain_label} samples ({skipped} skipped).")
    return dataset, metadata


# ============================================================
# SPLITTING
# ============================================================


def split_key_from_meta(meta: dict, mode: str) -> str:
    if mode == "sample":
        return f"sample_{meta['frame_idx']}"
    if mode == "initial_site":
        return meta["initial_site_key"]
    if mode == "final_basin":
        return meta["final_basin_key"]
    if mode == "cluster":
        return meta["group"]
    if mode == "initial_to_final":
        return meta["initial_to_final_key"]
    raise ValueError(f"Unknown SPLIT_MODE={mode!r}")


def grouped_stratified_split(
    metadata: List[dict],
    val_fraction: float,
    seed: int,
    mode: str,
):
    """Split whole groups, never individual samples, to reduce site leakage."""
    rng = np.random.default_rng(seed)

    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(metadata):
        group_to_indices[split_key_from_meta(meta, mode)].append(i)

    group_records = []
    for group_key, idxs in group_to_indices.items():
        ys = [metadata[i]["y_stable"] for i in idxs]
        init_motifs = [metadata[i]["motif_initial"] for i in idxs]
        final_motifs = [metadata[i]["motif_final"] for i in idxs]
        structural_groups = [metadata[i]["group"] for i in idxs]

        maj_y = Counter(ys).most_common(1)[0][0]
        maj_init = Counter(init_motifs).most_common(1)[0][0]
        maj_final = Counter(final_motifs).most_common(1)[0][0]
        maj_group = Counter(structural_groups).most_common(1)[0][0]

        strat = f"{maj_group}__init_{maj_init}__final_{maj_final}__y_{maj_y}"
        group_records.append((group_key, idxs, strat))

    strat_to_groups = defaultdict(list)
    for rec in group_records:
        strat_to_groups[rec[2]].append(rec)

    train_idx, val_idx = [], []

    for _strat, recs in strat_to_groups.items():
        recs = list(recs)
        rng.shuffle(recs)

        if len(recs) == 1:
            train_idx.extend(recs[0][1])
            continue

        n_val = max(1, int(round(len(recs) * val_fraction)))
        val_recs = recs[:n_val]
        train_recs = recs[n_val:]

        for _, idxs, _ in train_recs:
            train_idx.extend(idxs)
        for _, idxs, _ in val_recs:
            val_idx.extend(idxs)

    if len(val_idx) == 0:
        all_recs = list(group_records)
        rng.shuffle(all_recs)
        n_val_groups = max(1, int(round(len(all_recs) * val_fraction)))
        val_group_keys = {rec[0] for rec in all_recs[:n_val_groups]}

        train_idx, val_idx = [], []
        for group_key, idxs, _ in group_records:
            if group_key in val_group_keys:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx, group_to_indices


# ============================================================

# ============================================================
# MODEL: SHARED ENCODER + TWO BINARY DOMAIN HEADS
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
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.res_proj = (
            nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else None
        )

    def forward(self, x, edge_index, edge_attr):
        messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_res = self.res_proj(x) if self.res_proj is not None else x
        return x_res + self.node_mlp(torch.cat([x, messages], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        message = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return self.att_mlp(message) * message


class MotifRetentionTwoHeadGNN(nn.Module):
    """
    Shared message-passing/pooling encoder with one binary retention head per domain.

    The encoder never receives the domain as an input feature. domain_id is used
    only after pooling to route each graph embedding to the NP or SLAB head.

    Each head returns ONE logit. sigmoid(logit) = p(retained_exact).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.conv1 = EdgeGatedConv(node_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeGatedConv(hidden_dim, edge_dim, hidden_dim)
        self.conv3 = EdgeGatedConv(hidden_dim, edge_dim, hidden_dim)

        pooled_dim = 2 * hidden_dim

        self.np_retention_head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )
        self.slab_retention_head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )

    @staticmethod
    def weighted_mean_pool(x, weights, batch):
        weighted_sum = global_add_pool(x * weights, batch)
        weight_sum = global_add_pool(weights, batch)
        return weighted_sum / weight_sum.clamp_min(1e-8)

    def encode(self, x, edge_index, edge_attr, batch):
        x_in = x
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = F.relu(self.conv3(h, edge_index, edge_attr))

        is_o_anchor = x_in[:, 1:2]
        dist_to_o = x_in[:, 4:5]
        weights_o = torch.exp(-dist_to_o / POOL_SIGMA) + ANCHOR_BOOST * is_o_anchor
        g_o = self.weighted_mean_pool(h, weights_o, batch)

        is_h_anchor = x_in[:, 2:3]
        dist_to_h = x_in[:, 5:6]
        weights_h = torch.exp(-dist_to_h / POOL_SIGMA) + ANCHOR_BOOST * is_h_anchor
        g_h = self.weighted_mean_pool(h, weights_h, batch)

        return torch.cat([g_o, g_h], dim=-1)

    def forward(self, x, edge_index, edge_attr, batch, domain_id):
        embedding = self.encode(x, edge_index, edge_attr, batch)

        np_logits = self.np_retention_head(embedding).squeeze(-1)
        slab_logits = self.slab_retention_head(embedding).squeeze(-1)

        domain_id = domain_id.view(-1).long()
        is_np = domain_id == DOMAIN_TO_ID["NP"]
        logits = torch.where(is_np, np_logits, slab_logits)
        return logits


# ============================================================
# TRAINING / EVALUATION
# ============================================================


def binary_metrics(
    labels_np: np.ndarray,
    probs_np: np.ndarray,
    threshold: float = BINARY_THRESHOLD,
) -> dict:
    """Binary metrics with retained_exact=1 and migrated=0."""
    labels = labels_np.astype(int)
    probs = probs_np.astype(float)

    if len(labels) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision_retained": 0.0,
            "recall_retained": 0.0,
            "specificity_migrated": 0.0,
            "f1_retained": 0.0,
            "support_retained": 0,
            "support_migrated": 0,
            "predicted_retained": 0,
            "predicted_migrated": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "mean_p_retained": 0.0,
            "empirical_retained_fraction": 0.0,
        }

    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = float((preds == labels).mean())
    recall_retained = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_migrated = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = 0.5 * (recall_retained + specificity_migrated)
    precision_retained = tp / (tp + fp) if (tp + fp) else 0.0
    f1_retained = (
        2
        * precision_retained
        * recall_retained
        / (precision_retained + recall_retained)
        if (precision_retained + recall_retained)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy),
        "precision_retained": float(precision_retained),
        "recall_retained": float(recall_retained),
        "specificity_migrated": float(specificity_migrated),
        "f1_retained": float(f1_retained),
        "support_retained": int((labels == 1).sum()),
        "support_migrated": int((labels == 0).sum()),
        "predicted_retained": int((preds == 1).sum()),
        "predicted_migrated": int((preds == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "mean_p_retained": float(probs.mean()),
        "empirical_retained_fraction": float(labels.mean()),
    }


def run_epoch(model, loader, optimizer=None):
    """
    Binary BCE, calculated independently for each domain present in a batch.
    Present-domain losses are combined via DOMAIN_LOSS_WEIGHT before
    backpropagation, normalized by the weights of the domains actually
    present so a batch containing only one domain is unaffected (still gets
    that domain's own loss, undiluted).
    """
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_objective = 0.0
    n_batches = 0

    all_labels = []
    all_probs = []
    all_domains = []

    domain_loss_sum = defaultdict(float)
    domain_loss_count = defaultdict(int)

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(DEVICE)
            labels = batch.y_stable.view(-1).float()
            domains = batch.domain_id.view(-1).long()

            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                domains,
            )

            weighted_losses = []
            present_weight_sum = 0.0
            for domain_label, domain_id in DOMAIN_TO_ID.items():
                mask = domains == domain_id
                n_domain = int(mask.sum().item())
                if n_domain == 0:
                    continue

                domain_loss = F.binary_cross_entropy_with_logits(
                    logits[mask],
                    labels[mask],
                    reduction="mean",
                )
                weight = DOMAIN_LOSS_WEIGHT[domain_label]
                weighted_losses.append(weight * domain_loss)
                present_weight_sum += weight
                domain_loss_sum[domain_id] += domain_loss.item() * n_domain
                domain_loss_count[domain_id] += n_domain

            if not weighted_losses:
                raise RuntimeError("Batch unexpectedly contains no recognized domains.")

            loss = torch.stack(weighted_losses).sum() / present_weight_sum

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)

            total_objective += loss.item()
            n_batches += 1
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_domains.append(domains.detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    domains_np = np.concatenate(all_domains)

    metrics = binary_metrics(labels_np, probs_np, threshold=BINARY_THRESHOLD)
    metrics["loss"] = total_objective / max(1, n_batches)

    per_domain = {}
    for domain_label, domain_id in DOMAIN_TO_ID.items():
        mask = domains_np == domain_id
        if not np.any(mask):
            continue

        domain_metrics = binary_metrics(
            labels_np[mask],
            probs_np[mask],
            threshold=BINARY_THRESHOLD,
        )
        domain_metrics["loss"] = domain_loss_sum[domain_id] / max(
            1, domain_loss_count[domain_id]
        )
        domain_metrics["n"] = int(mask.sum())
        per_domain[domain_label] = domain_metrics

    metrics["per_domain"] = per_domain
    return metrics


def evaluate_per_motif_by_domain(model, loader) -> dict:
    """Detailed binary validation summary for every domain/initial-motif pair."""
    model.eval()
    rows = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            domains = batch.domain_id.view(-1).long()
            labels = batch.y_stable.view(-1).long()
            motifs = batch.y_motif_initial.view(-1).long()

            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                domains,
            )
            probs = torch.sigmoid(logits)

            for prob, label, motif_id, domain_id in zip(probs, labels, motifs, domains):
                domain = ID_TO_DOMAIN[int(domain_id.item())]
                motif = ID_TO_MOTIF.get(int(motif_id.item()), "UNK")
                rows[domain][motif].append((float(prob.item()), int(label.item())))

    out = {}
    for domain, motif_dict in sorted(rows.items()):
        out[domain] = {}
        for motif, values in sorted(motif_dict.items()):
            probs = np.array([v[0] for v in values], dtype=float)
            labels = np.array([v[1] for v in values], dtype=int)
            metrics = binary_metrics(
                labels,
                probs,
                threshold=BINARY_THRESHOLD,
            )
            out[domain][motif] = {
                "n": int(len(values)),
                **metrics,
            }
    return out


def domain_macro_value(metrics: dict, metric_name: str) -> float:
    """Average the selected validation metric across NP and SLAB."""
    values = []
    for domain_label in ["NP", "SLAB"]:
        if domain_label in metrics["per_domain"]:
            values.append(float(metrics["per_domain"][domain_label][metric_name]))

    if not values:
        raise RuntimeError("No domain metrics available for early stopping.")
    return float(np.mean(values))


def is_better(current_value: float, best_value: float, metric_name: str) -> bool:
    if metric_name == "loss":
        return current_value < best_value
    return current_value > best_value


# ============================================================
# DIAGNOSTICS / SUMMARIES
# ============================================================


def summarize_domain(metadata: List[dict], domain_label: str):
    print(f"\n=== {domain_label} DATASET SUMMARY ===")
    print(f"Samples: {len(metadata)}")

    retained_count = sum(meta["y_stable"] for meta in metadata)
    migrated_count = len(metadata) - retained_count
    print(
        f"Retained/migrated: {retained_count}/{migrated_count} "
        f"({retained_count / max(1, len(metadata)) * 100:.1f}% retained exact)"
    )

    print("Initial motif counts:")
    for motif, count in sorted(
        Counter(meta["motif_initial"] for meta in metadata).items()
    ):
        print(f"  {motif}: {count}")

    print("Final motif counts:")
    for motif, count in sorted(
        Counter(meta["motif_final"] for meta in metadata).items()
    ):
        print(f"  {motif}: {count}")

    print("Structural groups:")
    for group, count in sorted(Counter(meta["group"] for meta in metadata).items()):
        print(f"  {group}: {count}")


def build_domain_split(
    dataset,
    metadata,
    val_fraction: float,
    seed: int,
):
    """Generate a new grouped split. Used only when no fixed split file exists."""
    train_idx, val_idx, split_groups = grouped_stratified_split(
        metadata,
        val_fraction,
        seed=seed,
        mode=SPLIT_MODE,
    )

    if len(val_idx) == 0:
        raise RuntimeError(
            "Validation set is empty. Use SPLIT_MODE='sample' or reduce grouping strictness."
        )

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "split_groups": split_groups,
        "train_dataset": [dataset[i] for i in train_idx],
        "val_dataset": [dataset[i] for i in val_idx],
        "train_metadata": [metadata[i] for i in train_idx],
        "val_metadata": [metadata[i] for i in val_idx],
    }


def build_domain_split_from_fixed_frames(
    dataset,
    metadata,
    fixed_domain_info: dict,
    domain_label: str,
):
    """
    Reconstruct a saved split from source-frame indices.

    We deliberately do not trust saved dataset indices because a skipped frame can
    shift dataset indexing. frame_idx refers to the source extxyz frame identity
    and is therefore the safer key.
    """
    if (
        "train_frame_idx" not in fixed_domain_info
        or "val_frame_idx" not in fixed_domain_info
    ):
        raise ValueError(
            f"{domain_label}: fixed split does not contain train_frame_idx/val_frame_idx."
        )

    frame_to_dataset_idx = {}
    for dataset_idx, meta in enumerate(metadata):
        frame_idx = int(meta["frame_idx"])
        if frame_idx in frame_to_dataset_idx:
            raise ValueError(
                f"{domain_label}: duplicate frame_idx={frame_idx} in rebuilt dataset."
            )
        frame_to_dataset_idx[frame_idx] = dataset_idx

    train_frames = [int(v) for v in fixed_domain_info["train_frame_idx"]]
    val_frames = [int(v) for v in fixed_domain_info["val_frame_idx"]]

    train_frame_set = set(train_frames)
    val_frame_set = set(val_frames)

    overlap = train_frame_set & val_frame_set
    if overlap:
        raise ValueError(
            f"{domain_label}: fixed train/validation frame sets overlap: "
            f"{sorted(overlap)[:10]}"
        )

    saved_frames = train_frame_set | val_frame_set
    current_frames = set(frame_to_dataset_idx)

    missing_now = saved_frames - current_frames
    new_now = current_frames - saved_frames

    if missing_now or new_now:
        msg = [
            f"{domain_label}: current dataset does not exactly match the fixed split."
        ]
        if missing_now:
            msg.append(
                f"  Saved frames missing from current dataset: {sorted(missing_now)[:20]}"
            )
        if new_now:
            msg.append(
                f"  Current frames absent from saved split: {sorted(new_now)[:20]}"
            )
        msg.append(
            "Refusing to continue because architecture tests must use exactly the same split."
        )
        raise RuntimeError("\n".join(msg))

    train_idx = [frame_to_dataset_idx[frame_idx] for frame_idx in train_frames]
    val_idx = [frame_to_dataset_idx[frame_idx] for frame_idx in val_frames]

    split_groups = defaultdict(list)
    for i, meta in enumerate(metadata):
        split_groups[split_key_from_meta(meta, SPLIT_MODE)].append(i)

    train_keys = {split_key_from_meta(metadata[i], SPLIT_MODE) for i in train_idx}
    val_keys = {split_key_from_meta(metadata[i], SPLIT_MODE) for i in val_idx}
    key_overlap = train_keys & val_keys
    if key_overlap:
        raise RuntimeError(
            f"{domain_label}: fixed split has {len(key_overlap)} overlapping "
            f"{SPLIT_MODE} split keys between train and validation."
        )

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "split_groups": split_groups,
        "train_dataset": [dataset[i] for i in train_idx],
        "val_dataset": [dataset[i] for i in val_idx],
        "train_metadata": [metadata[i] for i in train_idx],
        "val_metadata": [metadata[i] for i in val_idx],
    }


def make_split_info(domain_data: dict) -> dict:
    """Serialize the currently active split using both dataset and frame indices."""
    split_info = {
        "split_mode": SPLIT_MODE,
        "split_seed_when_created": int(SPLIT_SEED),
        "domains": {},
    }

    for domain_label in ["NP", "SLAB"]:
        metadata = domain_data[domain_label]["metadata"]
        split = domain_data[domain_label]["split"]

        split_info["domains"][domain_label] = {
            "train_idx": split["train_idx"],
            "val_idx": split["val_idx"],
            "train_frame_idx": [metadata[i]["frame_idx"] for i in split["train_idx"]],
            "val_frame_idx": [metadata[i]["frame_idx"] for i in split["val_idx"]],
            "train_split_keys": sorted(
                {
                    split_key_from_meta(metadata[i], SPLIT_MODE)
                    for i in split["train_idx"]
                }
            ),
            "val_split_keys": sorted(
                {split_key_from_meta(metadata[i], SPLIT_MODE) for i in split["val_idx"]}
            ),
        }

    return split_info


def concat(parts):
    out = []
    for part in parts:
        out.extend(part)
    return out


# ============================================================
# MAIN
# ============================================================


def main():
    print(f"Device: {DEVICE}")
    print("Architecture: shared encoder + separate binary NP/SLAB retention heads")
    print("Target: retained_exact=1, migrated=0")
    print(f"Loss: BCEWithLogitsLoss, domain weights = {DOMAIN_LOSS_WEIGHT}")
    print(f"Decision threshold for reported metrics: {BINARY_THRESHOLD:.2f}")
    print(
        "Adsorption-state filter: exactly one O-H bond below "
        f"{OH_BOND_CUTOFF:.2f} Å and no O-H contact below "
        f"{MIN_OH_SEPARATION:.2f} Å"
    )
    if USE_HEADER_INITIAL_SITE:
        print("Initial site source: motif_label/o_anchor/h_anchor from extxyz headers")
    else:
        print("Initial site source: loose geometric cutoffs, maximum 4 anchors")
    print(f"Training seed: {TRAINING_SEED}")
    print(f"Fixed split mode: {USE_FIXED_SPLIT}")
    print(f"Fixed split file: {FIXED_SPLIT_FILE}")

    fixed_split_info = None
    fixed_split_loaded = False

    if USE_FIXED_SPLIT and os.path.exists(FIXED_SPLIT_FILE):
        with open(FIXED_SPLIT_FILE, "r") as handle:
            fixed_split_info = json.load(handle)

        saved_mode = fixed_split_info.get("split_mode")
        if saved_mode != SPLIT_MODE:
            raise RuntimeError(
                f"Fixed split was created with split_mode={saved_mode!r}, "
                f"but current SPLIT_MODE={SPLIT_MODE!r}."
            )

        for required_domain in ["NP", "SLAB"]:
            if required_domain not in fixed_split_info.get("domains", {}):
                raise RuntimeError(
                    f"Fixed split file is missing domain {required_domain!r}."
                )

        fixed_split_loaded = True
        print("Fixed split found: reusing saved train/validation frame identities.")
    elif USE_FIXED_SPLIT:
        print(
            f"Fixed split file not found. A new split will be generated ONCE "
            f"with SPLIT_SEED={SPLIT_SEED} and saved before training."
        )
    else:
        print(
            f"USE_FIXED_SPLIT=False: a new split will be generated with "
            f"SPLIT_SEED={SPLIT_SEED}."
        )

    domain_data = {}

    for domain_label in ["NP", "SLAB"]:
        initial_file, final_file = DOMAIN_FILES[domain_label]

        print(f"\n{'=' * 72}")
        print(f"LOADING {domain_label}")
        print(f"Initial: {initial_file}")
        print(f"Final:   {final_file}")

        initial_frames = read(initial_file, index=":")
        final_frames = read(final_file, index=":")
        print(
            f"Loaded {len(initial_frames)} initial and "
            f"{len(final_frames)} final frames."
        )

        group_labels = build_group_labels(initial_frames, domain_label)
        dataset, metadata = build_dataset(
            initial_frames,
            final_frames,
            group_labels,
            domain_label,
        )

        if len(dataset) == 0:
            raise RuntimeError(f"No usable {domain_label} samples were built.")

        summarize_domain(metadata, domain_label)

        val_fraction = VAL_FRACTION_NP if domain_label == "NP" else VAL_FRACTION_SLAB

        if fixed_split_loaded:
            split = build_domain_split_from_fixed_frames(
                dataset,
                metadata,
                fixed_split_info["domains"][domain_label],
                domain_label=domain_label,
            )
            split_source = "fixed frame identities"
        else:
            split = build_domain_split(
                dataset,
                metadata,
                val_fraction=val_fraction,
                seed=SPLIT_SEED,
            )
            split_source = f"generated with SPLIT_SEED={SPLIT_SEED}"

        train_retained = sum(
            int(data.y_stable.item()) for data in split["train_dataset"]
        )
        val_retained = sum(int(data.y_stable.item()) for data in split["val_dataset"])

        print(f"\n{domain_label} split mode: {SPLIT_MODE}")
        print(f"  Split source:  {split_source}")
        print(f"  Train samples: {len(split['train_dataset'])}")
        print(f"  Val samples:   {len(split['val_dataset'])}")
        print(f"  Split groups:  {len(split['split_groups'])}")
        print(
            f"  Train retained/migrated: {train_retained}/"
            f"{len(split['train_dataset']) - train_retained}"
        )
        print(
            f"  Val retained/migrated:   {val_retained}/"
            f"{len(split['val_dataset']) - val_retained}"
        )

        domain_data[domain_label] = {
            "dataset": dataset,
            "metadata": metadata,
            "initial_frames": initial_frames,
            "final_frames": final_frames,
            "group_labels": group_labels,
            "split": split,
        }

    # If this is the first run with the fixed-split workflow, freeze the split now,
    # before model training. Later architecture/seed runs will load these exact frames.
    if USE_FIXED_SPLIT and not fixed_split_loaded:
        fixed_split_info = make_split_info(domain_data)
        with open(FIXED_SPLIT_FILE, "w") as handle:
            json.dump(fixed_split_info, handle, indent=2)
        fixed_split_loaded = True
        print(f"\nCreated and froze split in: {FIXED_SPLIT_FILE}")

    train_dataset = concat(
        [
            domain_data["NP"]["split"]["train_dataset"],
            domain_data["SLAB"]["split"]["train_dataset"],
        ]
    )
    val_dataset = concat(
        [
            domain_data["NP"]["split"]["val_dataset"],
            domain_data["SLAB"]["split"]["val_dataset"],
        ]
    )

    train_metadata = concat(
        [
            domain_data["NP"]["split"]["train_metadata"],
            domain_data["SLAB"]["split"]["train_metadata"],
        ]
    )
    val_metadata = concat(
        [
            domain_data["NP"]["split"]["val_metadata"],
            domain_data["SLAB"]["split"]["val_metadata"],
        ]
    )

    print(f"\n{'=' * 72}")
    print("COMBINED TRAINING SET")
    print(f"Train total: {len(train_dataset)}")
    print(f"Val total:   {len(val_dataset)}")
    print("Train by domain:", dict(Counter(meta["domain"] for meta in train_metadata)))
    print("Val by domain:  ", dict(Counter(meta["domain"] for meta in val_metadata)))

    all_dataset = domain_data["NP"]["dataset"] + domain_data["SLAB"]["dataset"]
    node_dim = all_dataset[0].x.shape[1]
    edge_dim = all_dataset[0].edge_attr.shape[1]

    for data in all_dataset:
        if data.x.shape[1] != node_dim:
            raise ValueError("Node feature dimensions differ between samples/domains.")
        if data.edge_attr.shape[1] != edge_dim:
            raise ValueError("Edge feature dimensions differ between samples/domains.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MotifRetentionTwoHeadGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print("Heads: np_retention_head, slab_retention_head")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_value = float("inf") if EARLY_STOPPING_METRIC == "loss" else -float("inf")
    best_state = None
    patience_counter = 0

    print("\nStarting joint binary training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
        )

        val_domain_macro = domain_macro_value(
            val_metrics,
            EARLY_STOPPING_METRIC,
        )

        train_np = train_metrics["per_domain"].get("NP", {})
        train_slab = train_metrics["per_domain"].get("SLAB", {})
        val_np = val_metrics["per_domain"].get("NP", {})
        val_slab = val_metrics["per_domain"].get("SLAB", {})

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss={train_metrics['loss']:.4f} "
            f"bal_acc={train_metrics['balanced_accuracy']:.4f} | "
            f"NP={train_np.get('balanced_accuracy', 0.0):.4f} "
            f"SLAB={train_slab.get('balanced_accuracy', 0.0):.4f} || "
            f"Val loss={val_metrics['loss']:.4f} "
            f"bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"NP={val_np.get('balanced_accuracy', 0.0):.4f} "
            f"SLAB={val_slab.get('balanced_accuracy', 0.0):.4f} | "
            f"domain-macro-{EARLY_STOPPING_METRIC}={val_domain_macro:.4f}"
        )

        if is_better(val_domain_macro, best_value, EARLY_STOPPING_METRIC):
            best_value = val_domain_macro
            val_per_motif_by_domain = evaluate_per_motif_by_domain(
                model,
                val_loader,
            )

            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "task": "retained_exact_binary_two_domain_heads",
                "architecture": "shared_encoder_separate_NP_SLAB_binary_heads",
                "binary_label_definition": {
                    "0": "migrated",
                    "1": "retained_exact",
                },
                "binary_threshold": float(BINARY_THRESHOLD),
                "domain_to_id": DOMAIN_TO_ID,
                "id_to_domain": ID_TO_DOMAIN,
                "motif_to_id": MOTIF_TO_ID,
                "id_to_motif": ID_TO_MOTIF,
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "hidden_dim": HIDDEN_DIM,
                "split_mode": SPLIT_MODE,
                "early_stopping_metric": EARLY_STOPPING_METRIC,
                "best_domain_macro_value": float(best_value),
                "val_metrics": val_metrics,
                "val_per_motif_by_domain": val_per_motif_by_domain,
                "domain_splits": {
                    domain_label: {
                        "train_idx": domain_data[domain_label]["split"]["train_idx"],
                        "val_idx": domain_data[domain_label]["split"]["val_idx"],
                    }
                    for domain_label in ["NP", "SLAB"]
                },
                "metadata_by_domain": {
                    domain_label: domain_data[domain_label]["metadata"]
                    for domain_label in ["NP", "SLAB"]
                },
                "settings": {
                    "NP_INITIAL_EXTXYZ_FILE": NP_INITIAL_EXTXYZ_FILE,
                    "NP_FINAL_EXTXYZ_FILE": NP_FINAL_EXTXYZ_FILE,
                    "SLAB_INITIAL_EXTXYZ_FILE": SLAB_INITIAL_EXTXYZ_FILE,
                    "SLAB_FINAL_EXTXYZ_FILE": SLAB_FINAL_EXTXYZ_FILE,
                    "OH_BOND_CUTOFF": OH_BOND_CUTOFF,
                    "MIN_OH_SEPARATION": MIN_OH_SEPARATION,
                    "ADSORPTION_STATE_FILTER": "exactly_one_OH_bond",
                    "NI_O_CUTOFF_INITIAL": NI_O_CUTOFF_INITIAL,
                    "NI_H_CUTOFF_INITIAL": NI_H_CUTOFF_INITIAL,
                    "MAX_ANCHORS_PER_ADSORBATE": MAX_ANCHORS_PER_ADSORBATE,
                    "USE_HEADER_INITIAL_SITE": bool(USE_HEADER_INITIAL_SITE),
                    "INITIAL_SITE_SOURCE": (
                        "header_motif_and_anchors"
                        if USE_HEADER_INITIAL_SITE
                        else "geometry_loose_cutoffs_max4"
                    ),
                    "NI_O_CUTOFF_FINAL": NI_O_CUTOFF_FINAL,
                    "NI_O_FOLDED_CUTOFF_FINAL": NI_O_FOLDED_CUTOFF_FINAL,
                    "O_FOLDED_N_SHORT_FINAL": O_FOLDED_N_SHORT_FINAL,
                    "O_FOLDED_N_TOTAL_FINAL": O_FOLDED_N_TOTAL_FINAL,
                    "MAX_STANDARD_O_ANCHORS_FINAL": MAX_STANDARD_O_ANCHORS_FINAL,
                    "NI_H_CUTOFF_FINAL": NI_H_CUTOFF_FINAL,
                    "NI_H_FOLDED_CUTOFF_FINAL": NI_H_FOLDED_CUTOFF_FINAL,
                    "H_FOLDED_N_SHORT_FINAL": H_FOLDED_N_SHORT_FINAL,
                    "H_FOLDED_N_TOTAL_FINAL": H_FOLDED_N_TOTAL_FINAL,
                    "MAX_STANDARD_H_ANCHORS_FINAL": MAX_STANDARD_H_ANCHORS_FINAL,
                    "FINAL_ANCHOR_DEFINITION": "hybrid_at_least_2short_exactly_4extended_F",
                    "LOCAL_CUTOFF": LOCAL_CUTOFF,
                    "EDGE_CUTOFF": EDGE_CUTOFF,
                    "LOCAL_CUTOFF_BY_DOMAIN": LOCAL_CUTOFF_BY_DOMAIN,
                    "EDGE_CUTOFF_BY_DOMAIN": EDGE_CUTOFF_BY_DOMAIN,
                    "SPLIT_MODE": SPLIT_MODE,
                    "VAL_FRACTION_NP": VAL_FRACTION_NP,
                    "VAL_FRACTION_SLAB": VAL_FRACTION_SLAB,
                    "CN_FEATURE": "local_subgraph_PBC_aware",
                    "LOSS": "BCEWithLogitsLoss",
                    "DOMAIN_LOSS_WEIGHT": DOMAIN_LOSS_WEIGHT,
                    "CLASS_WEIGHTING": False,
                    "BINARY_THRESHOLD": float(BINARY_THRESHOLD),
                    "EARLY_STOPPING_METRIC": EARLY_STOPPING_METRIC,
                    "EARLY_STOPPING_DOMAIN_AGGREGATION": "mean_NP_SLAB",
                    "TRAINING_DOMAIN_LOSS_AGGREGATION": "weighted_present_domains_per_batch",
                    "DOMAIN_USED_AS_GRAPH_FEATURE": False,
                    "TRAINING_SEED": int(TRAINING_SEED),
                    "SPLIT_SEED": int(SPLIT_SEED),
                    "USE_FIXED_SPLIT": bool(USE_FIXED_SPLIT),
                    "FIXED_SPLIT_FILE": FIXED_SPLIT_FILE,
                },
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No best model saved.")

    torch.save(best_state, MODEL_OUT)
    print(
        f"\nBest validation domain-macro {EARLY_STOPPING_METRIC}: " f"{best_value:.6f}"
    )
    print(f"Saved model to: {MODEL_OUT}")

    print("\n=== BEST VALIDATION PERFORMANCE BY DOMAIN ===")
    for domain_label in ["NP", "SLAB"]:
        metrics = best_state["val_metrics"]["per_domain"].get(domain_label, {})
        if not metrics:
            continue
        print(
            f"{domain_label}: n={metrics.get('n', 0)} "
            f"loss={metrics.get('loss', 0.0):.4f} "
            f"acc={metrics.get('accuracy', 0.0):.4f} "
            f"bal_acc={metrics.get('balanced_accuracy', 0.0):.4f} "
            f"precision_ret={metrics.get('precision_retained', 0.0):.4f} "
            f"recall_ret={metrics.get('recall_retained', 0.0):.4f} "
            f"f1_ret={metrics.get('f1_retained', 0.0):.4f}"
        )

    print("\n=== BEST VALIDATION PERFORMANCE PER INITIAL MOTIF ===")
    for domain_label in ["NP", "SLAB"]:
        print(f"\n{domain_label}:")
        domain_results = best_state["val_per_motif_by_domain"].get(domain_label, {})
        for motif, row in domain_results.items():
            print(
                f"  {motif}: n={row.get('n', 0):4d} "
                f"emp_ret={row.get('empirical_retained_fraction', 0.0):.3f} "
                f"mean_p={row.get('mean_p_retained', 0.0):.3f} "
                f"acc={row.get('accuracy', 0.0):.3f} "
                f"bal_acc={row.get('balanced_accuracy', 0.0):.3f} "
                f"prec={row.get('precision_retained', 0.0):.3f} "
                f"rec={row.get('recall_retained', 0.0):.3f}"
            )

    # --------------------------------------------------------
    # Save stats
    # --------------------------------------------------------
    stats = {
        "architecture": "shared_encoder_separate_NP_SLAB_binary_heads",
        "task": "retained_exact_binary",
        "binary_label_definition": {"0": "migrated", "1": "retained_exact"},
        "binary_threshold": float(BINARY_THRESHOLD),
        "loss": "BCEWithLogitsLoss",
        "domain_loss_weight": DOMAIN_LOSS_WEIGHT,
        "class_weighting": False,
        "domain_to_id": DOMAIN_TO_ID,
        "adsorption_state_filter": {
            "required_state": "OH*+H*",
            "oh_bond_cutoff_A": float(OH_BOND_CUTOFF),
            "minimum_oh_separation_A": float(MIN_OH_SEPARATION),
            "applied_to": ["initial", "relaxed"],
        },
        "use_header_initial_site": bool(USE_HEADER_INITIAL_SITE),
        "initial_site_source": (
            "header_motif_and_anchors"
            if USE_HEADER_INITIAL_SITE
            else "geometry_loose_cutoffs_max4"
        ),
        "split_mode": SPLIT_MODE,
        "early_stopping_metric": EARLY_STOPPING_METRIC,
        "training_seed": int(TRAINING_SEED),
        "split_seed": int(SPLIT_SEED),
        "use_fixed_split": bool(USE_FIXED_SPLIT),
        "fixed_split_file": FIXED_SPLIT_FILE,
        "best_domain_macro_value": float(best_value),
        "val_metrics": best_state["val_metrics"],
        "val_per_motif_by_domain": best_state["val_per_motif_by_domain"],
        "domains": {},
    }

    for domain_label in ["NP", "SLAB"]:
        metadata = domain_data[domain_label]["metadata"]
        split = domain_data[domain_label]["split"]
        final_basin_counts = Counter(meta["final_basin_key"] for meta in metadata)

        changes = Counter()
        for meta in metadata:
            if meta["y_stable"] == UNSTABLE:
                if meta["same_motif"]:
                    key = (
                        f"{meta['motif_initial']}->{meta['motif_final']} "
                        "(same motif, different anchors)"
                    )
                else:
                    key = f"{meta['motif_initial']}->{meta['motif_final']}"
                changes[key] += 1

        n_retained = int(sum(meta["y_stable"] for meta in metadata))

        stats["domains"][domain_label] = {
            "n_frames": len(metadata),
            "n_retained_exact": n_retained,
            "n_migrated": int(len(metadata) - n_retained),
            "retained_fraction": float(n_retained / max(1, len(metadata))),
            "initial_motif_counts": dict(
                Counter(meta["motif_initial"] for meta in metadata)
            ),
            "final_motif_counts": dict(
                Counter(meta["motif_final"] for meta in metadata)
            ),
            "group_counts": dict(Counter(meta["group"] for meta in metadata)),
            "train_size": len(split["train_dataset"]),
            "val_size": len(split["val_dataset"]),
            "change_types": dict(changes),
            "n_unique_final_basins": len(final_basin_counts),
            "largest_final_basins": final_basin_counts.most_common(20),
        }

    with open(STATS_OUT, "w") as handle:
        json.dump(stats, handle, indent=2)
    print(f"Saved stats to: {STATS_OUT}")

    # Save the active split as a record for this run.
    split_info = make_split_info(domain_data)
    with open(SPLIT_OUT, "w") as handle:
        json.dump(split_info, handle, indent=2)
    print(f"Saved split info to: {SPLIT_OUT}")


if __name__ == "__main__":
    main()
