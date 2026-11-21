"""Train the nanoparticle-only M5 adsorption-energy model.

The adsorbate-containing extxyz dataset supplies the DFT adsorption-energy
targets and the relaxed OH*/H* geometries from which anchors are assigned.
Frames that no longer contain exactly one OH bond are discarded before graph
construction.
Graph coordinates are taken from the corresponding MACE-relaxed bare Ni
nanoparticles listed in ``INITIAL_NI_FILES``. The script writes the reproducible
train/validation split, best model checkpoint, and a JSON summary of the
dataset and validation metrics.

Run this script from its own directory so that the relative paths below resolve
as provided.
"""

import copy
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.io import read, write
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, MessagePassing

# ============================================================
# INPUTS, OUTPUTS, AND MODEL SETTINGS
# ============================================================

# Adsorbate-containing configurations and adsorption-energy targets.
EXTXYZ_FILE = "../datasets/dataset_NPs_M5_energy_model.extxyz"

# Generated split, checkpoint, and summary files.
TRAIN_EXTXYZ_OUT = "M5_energy_model_train_data.extxyz"
VAL_EXTXYZ_OUT = "M5_energy_model_valid_data.extxyz"
MODEL_OUT = "M5_energy_model.pt"
STATS_OUT = "M5_energy_model_dataset_stats.json"

# MACE-relaxed bare nanoparticles used to construct the input graphs. Dictionary
# keys must match the nanoparticle-size prefix in each frame's config_label.
INITIAL_NI_FILES = {
    "Ni25": "../ML_relaxed_bare_geom/MACE/training/Ni25_MACE_opt.xyz",
    "Ni30": "../ML_relaxed_bare_geom/MACE/training/Ni30_MACE_opt.xyz",
    "Ni35": "../ML_relaxed_bare_geom/MACE/training/Ni35_MACE_opt.xyz",
    "Ni40": "../ML_relaxed_bare_geom/MACE/training/Ni40_MACE_opt.xyz",
    "Ni45": "../ML_relaxed_bare_geom/MACE/training/Ni45_MACE_opt.xyz",
    "Ni50": "../ML_relaxed_bare_geom/MACE/training/Ni50_MACE_opt.xyz",
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

# --- Local graph built from the bare Ni nanoparticle ---
# Include Ni atoms within this distance of any O/H anchor.
LOCAL_CUTOFF = 5.5

# Connect Ni nodes separated by at most this distance.
EDGE_CUTOFF = 4.0

# --- distance basis expansion ---
N_RBF = 16
RBF_DMIN = 0.0
RBF_DMAX = 5.0
RBF_GAMMA = 10.0

# --- training ---
BATCH_SIZE = 32
LR = 8e-4
WEIGHT_DECAY = 1e-6
EPOCHS = 200
VAL_FRACTION = 0.20
EARLY_STOPPING_PATIENCE = 25
HIDDEN_DIM = 128

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

def count_ni_atoms(symbols: List[str]) -> int:
    return sum(1 for s in symbols if s == "Ni")


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


def find_adsorbate_indices(symbols: List[str], positions: np.ndarray) -> Tuple[int, int, int]:
    """Return the O, hydroxyl-H, and adsorbed-H indices.

    Each frame must contain one O, two H atoms, and exactly one H within
    ``OH_BOND_CUTOFF`` of O. This rejects both dissociated O*+H*+H* and
    molecular H2O-like configurations from the OH*+H* training state.
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
        d = float(np.linalg.norm(positions[h_idx] - o_pos))
        if d < OH_BOND_CUTOFF:
            oh_like.append((h_idx, d))

    if len(oh_like) == 1:
        h_oh_idx = oh_like[0][0]
        h_ads_idx = [h for h in h_indices if h != h_oh_idx][0]
        return o_idx, h_oh_idx, h_ads_idx

    distances = [float(np.linalg.norm(positions[h_idx] - o_pos)) for h_idx in h_indices]
    raise ValueError(
        f"Expected exactly one O-H distance below {OH_BOND_CUTOFF:.2f} Å, "
        f"found {len(oh_like)}; distances={distances}"
    )


def filter_target_adsorption_state(frames):
    """Keep only frames representing the target OH*+H* adsorption state."""
    kept = []
    rejected = []

    for input_idx, atoms in enumerate(frames):
        symbols, positions = get_symbols_positions(atoms)
        try:
            find_adsorbate_indices(symbols, positions)
        except ValueError as exc:
            rejected.append(
                (input_idx, str(atoms.info.get("config_label", "")), str(exc))
            )
        else:
            kept.append(atoms)

    print(f"Retained {len(kept)} OH*+H* frames; discarded {len(rejected)} invalid frames.")
    for input_idx, config_label, reason in rejected:
        print(f"  Discarded input frame {input_idx} ({config_label}): {reason}")

    return kept


def extract_o_anchor_indices_hybrid(
    ni_indices: List[int],
    positions: np.ndarray,
    o_idx: int,
) -> Tuple[List[int], str, Dict[str, object]]:
    """Extract O anchors using the hybrid T/B/H-versus-F definition.

    Standard T/B/H sites use Ni-O contacts below ``NI_O_CUTOFF`` and retain
    at most three anchors.

    An F site requires at least ``O_FOLDED_N_SHORT`` short contacts and exactly
    ``O_FOLDED_N_TOTAL`` contacts below ``NI_O_FOLDED_CUTOFF``. All four
    extended-cutoff contacts are then retained.
    """
    o_pos = positions[o_idx]
    extended_hits = []

    for ni_idx in ni_indices:
        d = float(np.linalg.norm(positions[ni_idx] - o_pos))
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
        rule = "folded_2short_plus_2extended"
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
) -> Tuple[List[int], str, Dict[str, object]]:
    """Extract H anchors using the hybrid T/B/H-versus-F definition.

    Standard T/B/H sites use Ni-H contacts below ``NI_H_CUTOFF`` and retain
    at most three anchors.

    An F site requires at least ``H_FOLDED_N_SHORT`` short contacts and exactly
    ``H_FOLDED_N_TOTAL`` contacts below ``NI_H_FOLDED_CUTOFF``. All four
    extended-cutoff contacts are then retained.
    """
    h_pos = positions[h_ads_idx]
    extended_hits = []

    for ni_idx in ni_indices:
        d = float(np.linalg.norm(positions[ni_idx] - h_pos))
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
        rule = "folded_2short_plus_2extended"
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
    cutoff: float = LOCAL_CUTOFF,
) -> List[int]:
    if len(anchor_indices) == 0:
        raise ValueError("No anchors found; cannot build local environment.")

    keep = set(anchor_indices)
    anchor_positions = positions[anchor_indices]

    for ni_idx in ni_indices:
        ni_pos = positions[ni_idx]
        dmin = np.min(np.linalg.norm(anchor_positions - ni_pos[None, :], axis=1))
        if dmin <= cutoff:
            keep.add(ni_idx)

    return sorted(keep)


def compute_coordination_numbers(
    ni_positions: np.ndarray,
    cutoff: float = EDGE_CUTOFF,
) -> np.ndarray:
    n = len(ni_positions)
    cn = np.zeros(n, dtype=np.float32)
    for i in range(n):
        count = 0
        for j in range(n):
            if i == j:
                continue
            d = np.linalg.norm(ni_positions[i] - ni_positions[j])
            if d <= cutoff:
                count += 1
        cn[i] = float(count)
    return cn


def build_node_features(
    keep_indices: List[int],
    positions: np.ndarray,
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
) -> np.ndarray:
    """Build the eight node features used by the bare-Ni graph.

    Features are a Ni indicator; O-, H-, and any-anchor indicators; distances
    to the O, H, and combined anchor centroids; and local Ni coordination.
    """
    local_positions = positions[keep_indices]
    cn = compute_coordination_numbers(local_positions, cutoff=EDGE_CUTOFF)

    o_set = set(o_anchor_idx)
    h_set = set(h_anchor_idx)
    any_set = o_set | h_set

    o_center = positions[o_anchor_idx].mean(axis=0)
    h_center = positions[h_anchor_idx].mean(axis=0)
    any_center = positions[sorted(any_set)].mean(axis=0)

    feats = []
    for local_i, global_idx in enumerate(keep_indices):
        pos = positions[global_idx]
        feats.append([
            1.0,
            1.0 if global_idx in o_set else 0.0,
            1.0 if global_idx in h_set else 0.0,
            1.0 if global_idx in any_set else 0.0,
            float(np.linalg.norm(pos - o_center)),
            float(np.linalg.norm(pos - h_center)),
            float(np.linalg.norm(pos - any_center)),
            float(cn[local_i]),
        ])

    return np.asarray(feats, dtype=np.float32)


def build_edges_and_edge_attr(local_positions: np.ndarray, cutoff: float = EDGE_CUTOFF):
    """Build directed Ni-Ni edges and Gaussian-expanded distance features."""
    edge_src = []
    edge_dst = []
    distances = []

    n = len(local_positions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = np.linalg.norm(local_positions[i] - local_positions[j])
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


def build_group_labels(frames) -> List[str]:
    """Extract the nanoparticle size label from each frame's config_label."""
    import re as _re
    labels = []
    for atoms in frames:
        config_label = atoms.info.get("config_label", "")
        m = _re.match(r"(Ni\d+)_", config_label)
        labels.append(m.group(1) if m else "unknown")
    return labels


def stratified_split(labels: List[str], val_fraction: float, seed: int = 42):
    """Create a reproducible split while preserving every supplied label."""
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


def make_combined_strat_labels(group_labels: List[str], motif_labels: List[str]) -> List[str]:
    return [f"{g}__{m}" for g, m in zip(group_labels, motif_labels)]


def summarize_counter(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}

# ============================================================
# DATASET BUILDING
# ============================================================

def process_frame(atoms, initial_refs, group_label: str, frame_idx: int):
    """Convert one relaxed OH*+H* structure into a bare-site graph."""
    symbols, positions = get_symbols_positions(atoms)
    n_ni = count_ni_atoms(symbols)
    e_ads = get_target_energy(atoms)

    ni_indices = [i for i, s in enumerate(symbols) if s == "Ni"]
    o_idx, h_oh_idx, h_ads_idx = find_adsorbate_indices(symbols, positions)

    o_anchor_idx, o_site, o_anchor_meta = extract_o_anchor_indices_hybrid(
        ni_indices,
        positions,
        o_idx,
    )
    h_anchor_idx, h_site, h_anchor_meta = extract_h_anchor_indices_hybrid(
        ni_indices,
        positions,
        h_ads_idx,
    )

    # Anchor identities come from the relaxed adsorbate structure above. The
    # same Ni indices are mapped onto the corresponding MACE-relaxed bare NP,
    # which supplies every coordinate used by the graph.
    if group_label not in initial_refs:
        raise ValueError(
            f"Frame {frame_idx}: no bare reference found for group "
            f"{group_label!r}. Available groups: {sorted(initial_refs)}"
        )

    initial_atoms = initial_refs[group_label]
    initial_symbols, positions_initial = get_symbols_positions(initial_atoms)

    initial_ni_indices = [i for i, s in enumerate(initial_symbols) if s == "Ni"]

    if len(initial_ni_indices) != n_ni:
        raise ValueError(
            f"Frame {frame_idx}: initial reference {group_label} has wrong number of Ni atoms."
        )

    if len(o_anchor_idx) == 0:
        raise ValueError(f"Frame {frame_idx}: no O anchors found with NI_O_CUTOFF={NI_O_CUTOFF}")
    if len(h_anchor_idx) == 0:
        raise ValueError(f"Frame {frame_idx}: no H anchors found with NI_H_CUTOFF={NI_H_CUTOFF}")

    motif = o_site + h_site
    motif_id = MOTIF_TO_ID[motif]

    all_anchor_idx = sorted(set(o_anchor_idx) | set(h_anchor_idx))
    keep_ni = select_bare_local_environment(
        initial_ni_indices,
        positions_initial,
        all_anchor_idx,
        LOCAL_CUTOFF,
    )

    local_positions = positions_initial[keep_ni]
    x = build_node_features(keep_ni, positions_initial, o_anchor_idx, h_anchor_idx)
    edge_index, edge_attr = build_edges_and_edge_attr(local_positions, EDGE_CUTOFF)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=torch.tensor(local_positions, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_energy=torch.tensor([e_ads], dtype=torch.float),
        y_motif=torch.tensor([motif_id], dtype=torch.long),
        frame_idx=torch.tensor([frame_idx], dtype=torch.long),
        n_ni=torch.tensor([n_ni], dtype=torch.long),
        num_nodes=torch.tensor([len(keep_ni)], dtype=torch.long),
    )

    meta = {
        "frame_idx": frame_idx,
        "n_ni": n_ni,
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
        "input_group_label": group_label,
    }
    return data, meta

# ============================================================
# BARE REFERENCE STRUCTURES
# ============================================================

def load_initial_reference_structures():
    """Load and validate the bare Ni structures listed in INITIAL_NI_FILES."""
    refs = {}

    for group_label, filename in INITIAL_NI_FILES.items():
        atoms = read(filename)
        symbols = atoms.get_chemical_symbols()

        if any(s != "Ni" for s in symbols):
            raise ValueError(f"{group_label} reference contains non-Ni atoms.")

        refs[group_label] = atoms

    return refs

def build_pyg_dataset(extxyz_file: str, initial_refs):
    """Read all frames and construct their PyTorch Geometric graphs."""
    loaded_frames = read(extxyz_file, index=":")
    print(f"Loaded {len(loaded_frames)} frames from {extxyz_file}")
    frames = filter_target_adsorption_state(loaded_frames)

    group_labels = build_group_labels(frames)

    dataset = []
    metadata = []
    motif_labels = []

    for frame_idx, atoms in enumerate(frames):
        group_label = group_labels[frame_idx]
        data, meta = process_frame(atoms, initial_refs, group_label, frame_idx)
        dataset.append(data)
        metadata.append(meta)
        motif_labels.append(meta["motif"])

    return dataset, metadata, frames, group_labels, motif_labels

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

        self.res_proj = None
        if node_dim != hidden_dim:
            self.res_proj = nn.Linear(node_dim, hidden_dim)

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

# ============================================================
# MODEL
# ============================================================

class BareSiteGCNN(nn.Module):
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

        # Concatenate global, O-focused, and H-focused graph embeddings.
        pooled_dim = 3 * hidden_dim

        self.energy_head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )


    @staticmethod
    def weighted_mean_pool(x: torch.Tensor, weights: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        weighted_sum = global_add_pool(x * weights, batch)
        weight_sum = global_add_pool(weights, batch)
        return weighted_sum / weight_sum.clamp_min(1e-8)

    def forward(self, x, edge_index, edge_attr, batch):
        # Preserve the input features used to construct pooling weights.
        x_in = x

        h = self.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv3(h, edge_index, edge_attr)
        h = F.relu(h)

        # Global mean pooling summarizes the complete local environment.
        g_global = global_mean_pool(h, batch)

        # O-focused pooling emphasizes O anchors and nearby Ni atoms.
        is_o_anchor = x_in[:, 1:2]
        dist_to_o = x_in[:, 4:5]

        locality_o = torch.exp(-dist_to_o / POOL_SIGMA)
        weights_o = locality_o + ANCHOR_BOOST * is_o_anchor
        g_o = self.weighted_mean_pool(h, weights_o, batch)

        # H-focused pooling applies the same construction around H*.
        is_h_anchor = x_in[:, 2:3]
        dist_to_h = x_in[:, 5:6]

        locality_h = torch.exp(-dist_to_h / POOL_SIGMA)
        weights_h = locality_h + ANCHOR_BOOST * is_h_anchor
        g_h = self.weighted_mean_pool(h, weights_h, batch)

        g = torch.cat([g_global, g_o, g_h], dim=-1)

        e_pred = self.energy_head(g).squeeze(-1)
        return e_pred


# ============================================================
# TRAINING
# ============================================================

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_reg_loss = 0.0
    total_mae = 0.0
    n_samples = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(DEVICE)

            e_pred = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            e_true = batch.y_energy.view(-1)

            reg_loss = F.mse_loss(e_pred, e_true)
            loss = REG_LOSS_WEIGHT * reg_loss

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = e_true.size(0)
            total_loss += loss.item() * bs
            total_reg_loss += reg_loss.item() * bs
            total_mae += mae(e_pred, e_true).item() * bs
            n_samples += bs

    return {
        "loss": total_loss / n_samples,
        "reg_loss": total_reg_loss / n_samples,
        "mae": total_mae / n_samples,
    }


def evaluate_per_motif(model, loader):
    model.eval()
    motif_errors = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            e_pred = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            e_true = batch.y_energy.view(-1)
            m_true = batch.y_motif.view(-1)

            for ep, et, mt in zip(e_pred, e_true, m_true):
                motif = ID_TO_MOTIF[int(mt.item())]
                motif_errors[motif].append(abs(float(ep.item()) - float(et.item())))

    out = {}
    for motif in sorted(motif_errors.keys()):
        out[motif] = {
            "count": len(motif_errors[motif]),
            "mae_eV": float(np.mean(motif_errors[motif])),
        }
    return out

# ============================================================
# MAIN
# ============================================================

def main():

    initial_refs = load_initial_reference_structures()

    # Build the complete dataset from the input extxyz file.
    dataset, metadata, frames, group_labels, motif_labels = build_pyg_dataset(
        EXTXYZ_FILE,
        initial_refs,
    )

    print("\nMotif counts:")
    motif_counts = Counter(motif_labels)
    for motif, count in sorted(motif_counts.items()):
        print(f"  {motif}: {count}")

    print("\nO-anchor counts:")
    o_counts = Counter(meta["o_anchor_count"] for meta in metadata)
    for n, count in sorted(o_counts.items()):
        print(f"  {n}: {count}")

    print("\nH-anchor counts:")
    h_counts = Counter(meta["h_anchor_count"] for meta in metadata)
    for n, count in sorted(h_counts.items()):
        print(f"  {n}: {count}")

    print("\nO-anchor extraction rules:")
    o_rule_counts = Counter(meta["o_anchor_rule"] for meta in metadata)
    for rule, count in sorted(o_rule_counts.items()):
        print(f"  {rule}: {count}")

    print("\nH-anchor extraction rules:")
    h_rule_counts = Counter(meta["h_anchor_rule"] for meta in metadata)
    for rule, count in sorted(h_rule_counts.items()):
        print(f"  {rule}: {count}")

    # Stratify by both nanoparticle size and final OH/H motif.
    combined_labels = make_combined_strat_labels(group_labels, motif_labels)
    combined_counts = Counter(combined_labels)

    print("\nCombined stratification groups:")
    for label, count in sorted(combined_counts.items()):
        print(f"  {label}: {count}")

    train_idx, val_idx = stratified_split(
        combined_labels,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]

    train_frames = [frames[i] for i in train_idx]
    val_frames = [frames[i] for i in val_idx]

    # Persist the exact split for reproducible training and evaluation.
    write(TRAIN_EXTXYZ_OUT, train_frames, format="extxyz")
    write(VAL_EXTXYZ_OUT, val_frames, format="extxyz")

    print("\nGenerated stratified split:")
    print(f"  Input dataset: {EXTXYZ_FILE}")
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size:   {len(val_dataset)}")
    print(f"  Saved train split to: {TRAIN_EXTXYZ_OUT}")
    print(f"  Saved val split to:   {VAL_EXTXYZ_OUT}")

    # Dedicated RNG for reproducible minibatch ordering.
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    node_dim = dataset[0].x.shape[1]
    edge_dim = dataset[0].edge_attr.shape[1]

    model = BareSiteGCNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nHidden dimension: {HIDDEN_DIM}")
    print(f"Total model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_score = float("inf")
    best_state = None
    patience_counter = 0

    print("\nStarting training...\n")
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, optimizer=None)

        val_score = val_metrics["mae"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train MAE: {train_metrics['mae']:.6f} eV | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val MAE: {val_metrics['mae']:.6f} eV"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "hidden_dim": HIDDEN_DIM,
                "n_parameters": int(n_params),
                "energy_only": True,
                "motif_to_id": MOTIF_TO_ID,
                "id_to_motif": ID_TO_MOTIF,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "group_labels": group_labels,
                "motif_labels": motif_labels,
                "combined_labels": combined_labels,
                "settings": {
                    "EXTXYZ_FILE": EXTXYZ_FILE,
                    "TRAIN_EXTXYZ_OUT": TRAIN_EXTXYZ_OUT,
                    "VAL_EXTXYZ_OUT": VAL_EXTXYZ_OUT,
                    "NI_O_CUTOFF": NI_O_CUTOFF,
                    "NI_O_FOLDED_CUTOFF": NI_O_FOLDED_CUTOFF,
                    "O_FOLDED_N_SHORT": O_FOLDED_N_SHORT,
                    "O_FOLDED_N_TOTAL": O_FOLDED_N_TOTAL,
                    "MAX_STANDARD_O_ANCHORS": MAX_STANDARD_O_ANCHORS,
                    "O_ANCHOR_DEFINITION": "hybrid_2p20_standard_atleast2short_4total_F",
                    "NI_H_CUTOFF": NI_H_CUTOFF,
                    "NI_H_FOLDED_CUTOFF": NI_H_FOLDED_CUTOFF,
                    "H_FOLDED_N_SHORT": H_FOLDED_N_SHORT,
                    "H_FOLDED_N_TOTAL": H_FOLDED_N_TOTAL,
                    "OH_BOND_CUTOFF": OH_BOND_CUTOFF,
                    "MAX_STANDARD_H_ANCHORS": MAX_STANDARD_H_ANCHORS,
                    "H_ANCHOR_DEFINITION": "hybrid_2p00_standard_atleast2short_4total_F",
                    "SITE_TYPES": list(SITE_TYPES),
                    "LOCAL_CUTOFF": LOCAL_CUTOFF,
                    "EDGE_CUTOFF": EDGE_CUTOFF,
                    "N_RBF": N_RBF,
                    "RBF_DMIN": RBF_DMIN,
                    "RBF_DMAX": RBF_DMAX,
                    "RBF_GAMMA": RBF_GAMMA,
                    "BATCH_SIZE": BATCH_SIZE,
                    "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "SEED": SEED,
                    "VAL_FRACTION": VAL_FRACTION,
                    "SPLIT_DEFINITION": "stratified_by_np_size_and_motif",
                    "REG_LOSS_WEIGHT": REG_LOSS_WEIGHT,
                    "TRAINING_OBJECTIVE": "energy_only",
                },
                "best_val_score": best_val_score,
                "best_val_mae_eV": val_metrics["mae"],
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_per_motif": evaluate_per_motif(model, val_loader),
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
        "n_frames": len(frames),
        "motif_counts": summarize_counter(motif_counts),
        "o_anchor_counts": summarize_counter(o_counts),
        "o_anchor_rule_counts": summarize_counter(o_rule_counts),
        "h_anchor_counts": summarize_counter(h_counts),
        "h_anchor_rule_counts": summarize_counter(h_rule_counts),
        "combined_label_counts": summarize_counter(combined_counts),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
        "split_definition": "stratified_by_np_size_and_motif",
        "train_extxyz_out": TRAIN_EXTXYZ_OUT,
        "val_extxyz_out": VAL_EXTXYZ_OUT,
        "seed": int(SEED),
        "hidden_dim": int(HIDDEN_DIM),
        "n_parameters": int(n_params),
        "training_objective": "energy_only",
        "best_val_mae_eV": float(best_state["best_val_mae_eV"]),
        "val_per_motif": best_state["val_per_motif"],
        "example_metadata_first5": metadata[:5],
    }

    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved best model to: {MODEL_OUT}")
    print(f"Saved dataset statistics to: {STATS_OUT}")
    print(f"Best validation MAE: {best_state['best_val_mae_eV']:.6f} eV")


if __name__ == "__main__":
    main()
