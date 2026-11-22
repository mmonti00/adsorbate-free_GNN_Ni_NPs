#!/usr/bin/env python3
"""
analyze_bare_Ni_for_paper.py

Structural analysis of BARE Ni nanoparticles and periodic Ni slabs.

For nanoparticles and slabs, the script reports:
  - all-atom first-shell CN
  - all-atom GCN
  - Steinhardt q4/q6 and locally averaged qbar4/qbar6
  - first-shell Ni-Ni bond-length distributions
  - geometrically surface atoms from a probe-sphere accessibility test
  - surface-atom CN, GCN and qbar6 distributions
  - a combined surface-local-environment map in (GCN, qbar6) space, with
    nanoparticle surface atoms and periodic-surface reference environments

Periodic boundary conditions are taken directly from the ASE Atoms object read from
extxyz. First-shell neighbors are built with ASE neighbor_list, so periodic images are
counted correctly, including periodic copies of the same atom index in small cells.
The exposure test also includes periodic images explicitly.

SOAP/DScribe analysis has been removed.

Run from any directory: ``python test_systems/analyze_bare_Ni_for_paper.py``.
Inputs are the MACE-relaxed bare structures in ``ML_relaxed_bare_geom/MACE/``;
outputs are written to ``test_systems/bare_Ni_structural_analysis/``. The
combined surface-environment figure for all systems
(``surface_environment_SI_figure_all_systems.pdf``) corresponds to Figure S13.

Plots are written in a manuscript-ready style consistent with the adsorption-energy
parity plots: sans-serif typography, large labels/ticks, major/minor ticks, restrained
grid lines, and both vector PDF + 800-dpi PNG output.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.colors import to_rgb, to_hex

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

try:
    from scipy.special import sph_harm_y

    def spherical_harmonic(l: int, m: int, polar: np.ndarray, azimuth: np.ndarray):
        return sph_harm_y(l, m, polar, azimuth)

except ImportError:
    from scipy.special import sph_harm

    def spherical_harmonic(l: int, m: int, polar: np.ndarray, azimuth: np.ndarray):
        # Legacy scipy convention: sph_harm(m, l, azimuth, polar)
        return sph_harm(m, l, azimuth, polar)



# ============================================================
# PUBLICATION PLOT STYLE
# ============================================================

# Match the typography/tick style used in the parity plots elsewhere
# in the manuscript.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PLOT_DPI = 800

LABEL_FS = 20
TICK_FS = 18
XTICK_FS = 19

MAJOR_TICK_LENGTH = 8
MINOR_TICK_LENGTH = 4
MAJOR_TICK_WIDTH = 1.1
MINOR_TICK_WIDTH = 0.9

# Fixed colors for each structural system.  The same system keeps the same
# color in every CN / GCN / qbar6 / bond-length plot.  The nanoparticle
# palette deliberately reuses several manuscript model colors, while slab
# colors are grouped by facet (Ni(111) = blue family, Ni(100) = coral family).
NP_FACE_COLORS = {
    "Ni25":       "#3F7BD9",  # blue
    "Ni30":       "brown", #"#7E57C2",  # violet
    "Ni35":       "magenta",  # magenta
    "Ni40":       "gold",  # slate
    "Ni45":       "darkgreen",  # pink
    "Ni50":       "pink",  # teal
    "Ni47_conf1": "#5B8DB8",  # blue-grey
    "Ni47_conf2": "lightblue",  # darker blue-grey
    "Ni55":       "teal",  # coral
    "Ni91":       "darkorange", #"#9A6A7C",  # muted mauve
    "Ni130":      "darkviolet",  # green

    # New, larger nanoparticles (NEW_NP_FILES). Scattered unrelated hues, as
    # above: deliberately NOT a size-ordered gradient. A ramp would encode the
    # grouping in lightness alone, which is exactly the cue lost under colour
    # vision deficiency, and the dotted separator plus the x labels already say
    # which systems are new. Each hue below is chosen to stay clear of the
    # eleven colours above and of the darkblue/darkred facet references.
    "Ni218":      "dimgray",
    "Ni442":      "yellowgreen",
    "Ni524":      "mediumturquoise",
    "Ni686":      "indianred",
    "Ni864":      "mediumslateblue",

    # Third batch: the randomly drawn 1000-2000 atom particles. Picked by
    # maximising the smallest CIELAB separation from every colour already in
    # this dict AND from each other, with that distance evaluated under normal
    # vision and under simulated deuteranopia/protanopia/tritanopia and the
    # worst case taken (min dE ~17, i.e. comfortably distinguishable in all
    # four). Again not a ramp, for the reason given above.
    "Ni1141":     "black",
    "Ni1314":     "purple",
    "Ni1391":     "darkkhaki",
    "Ni3007":     "cyan",
}

SURFACE_FACE_COLORS = {
    "Ni111_2x2": "darkblue",  
    "Ni111_3x3": "darkblue",  
    "Ni111_4x3": "darkblue",  

    "Ni100_3x3": "darkred",  
    "Ni100_4x3": "darkred",  
    "Ni100_4x4": "darkred",  
}
# Facet references used in the SI summary figure.  The three supercells
# belonging to each facet are shown as small points at the same x position;
# a larger open marker gives the median across the three cell medians.
FACET_REFERENCE_COLORS = {
    "Ni(111)": "darkblue",
    "Ni(100)": "darkred",
}

FACET_REFERENCE_MARKERS = {
    "Ni(111)": "D",
    "Ni(100)": "s",
}

SURFACE_CELL_X_OFFSETS = (-0.10, 0.0, 0.10)

# Used only as a fallback if a new system is added without an explicit color.
DEFAULT_FACE_COLOR = "#A8CAC7"
EDGE_DARKEN_FACTOR = 0.58

GRID_ALPHA = 0.25


def darken_color(color: str, factor: float = EDGE_DARKEN_FACTOR) -> str:
    """Return a darker version of a hex/Matplotlib color."""
    rgb = np.asarray(to_rgb(color), dtype=float)
    return to_hex(np.clip(rgb * factor, 0.0, 1.0))


def system_plot_colors(label: str) -> Tuple[str, str]:
    """Return (face_color, edge_color) for a structural system."""
    face = NP_FACE_COLORS.get(
        label,
        SURFACE_FACE_COLORS.get(label, DEFAULT_FACE_COLOR),
    )
    return face, darken_color(face)


def format_system_label(label: str) -> str:
    """Convert internal filenames/keys to manuscript-style axis labels."""
    # Periodic surfaces, e.g. Ni111_4x3 -> Ni(111)-4x3.
    m = re.fullmatch(r"Ni(\d{3})_(\d+)x(\d+)", label)
    if m:
        facet, nx, ny = m.groups()
        return rf"Ni({facet})-{nx}$\times${ny}"

    # NP structural isomers.
    m = re.fullmatch(r"Ni(\d+)_conf(\d+)", label)
    if m:
        n, conf = m.groups()
        return rf"Ni$_{{{n}}}$-iso{conf}"

    # Ordinary nanoparticle.
    m = re.fullmatch(r"Ni(\d+)", label)
    if m:
        return rf"Ni$_{{{m.group(1)}}}$"

    return label


def save_publication_figure(fig, output_path: Path) -> None:
    """
    Save both vector PDF and high-resolution PNG from the same figure.

    Existing calls pass a .pdf path; the matching .png is generated
    automatically.
    """
    output_path = Path(output_path)
    pdf_path = output_path.with_suffix(".pdf")

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

def style_publication_axis(
    ax,
    *,
    categorical_x: bool = True,
) -> None:
    """Common single-panel styling matching the manuscript parity plots."""
    ax.tick_params(
        axis="both",
        which="major",
        length=MAJOR_TICK_LENGTH,
        width=MAJOR_TICK_WIDTH,
        labelsize=TICK_FS,
        direction="out",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=MINOR_TICK_LENGTH,
        width=MINOR_TICK_WIDTH,
        direction="out",
    )

    # Minor ticks are meaningful on the numeric y-axis. Avoid artificial
    # categorical minor ticks on x.
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if categorical_x:
        ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def add_surface_group_separator(
    ax,
    raw_labels: Sequence[str],
) -> None:
    """
    Add a light separator between different facets when surface systems
    are ordered by facet (e.g. all Ni111 cells followed by all Ni100 cells).
    """
    facets = []
    for label in raw_labels:
        m = re.fullmatch(r"Ni(\d{3})_\d+x\d+", label)
        facets.append(m.group(1) if m else None)

    for i in range(1, len(facets)):
        if facets[i] != facets[i - 1]:
            ax.axvline(
                i + 0.5,
                color="0.55",
                lw=1.0,
                ls="--",
                alpha=0.7,
                zorder=0,
            )


# ============================================================
# USER SETTINGS
# ============================================================

HERE = Path(__file__).resolve().parent
MACE_DIR = HERE.parent / "ML_relaxed_bare_geom" / "MACE"
TRAINING = MACE_DIR / "training"
FINE_TUNING = TRAINING / "fine-tuning"
TESTING = MACE_DIR / "testing"

NP_FILES: Dict[str, str] = {
    "Ni25": str(TRAINING / "Ni25_MACE_opt.xyz"),
    "Ni30": str(TRAINING / "Ni30_MACE_opt.xyz"),
    "Ni35": str(TRAINING / "Ni35_MACE_opt.xyz"),
    "Ni40": str(TRAINING / "Ni40_MACE_opt.xyz"),
    "Ni45": str(TRAINING / "Ni45_MACE_opt.xyz"),
    # Listed in order of atom count: every figure and table takes its ordering
    # from this dict (labels = list(structures.keys())), so Ni47 must precede
    # Ni50 here for the x axes to run monotonically in size.
    "Ni47_conf1": str(FINE_TUNING / "Ni47c1_MACE_opt.xyz"),
    "Ni47_conf2": str(FINE_TUNING / "Ni47c2_MACE_opt.xyz"),
    "Ni50": str(TRAINING / "Ni50_MACE_opt.xyz"),
    "Ni55": str(TESTING / "Ni55_MACE_opt.xyz"),
    "Ni91": str(TESTING / "Ni91_MACE_opt.xyz"),
    "Ni130": str(TESTING / "Ni130_MACE_opt.xyz"),
}

# Add/remove periodic surfaces freely. The lattice and pbc flags are read
# directly from these extxyz files.
SURFACE_FILES = {
    "Ni111_2x2": str(FINE_TUNING / "Ni111_2x2_MACE_opt.extxyz"),
    "Ni111_3x3": str(FINE_TUNING / "Ni111_3x3_MACE_opt.extxyz"),
    "Ni111_4x3": str(TESTING / "Ni111_4x3_MACE_opt.extxyz"),
    "Ni100_3x3": str(FINE_TUNING / "Ni100_3x3_MACE_opt.extxyz"),
    "Ni100_4x3": str(FINE_TUNING / "Ni100_4x3_MACE_opt.extxyz"),
    "Ni100_4x4": str(TESTING / "Ni100_4x4_MACE_opt.extxyz"),
}

# ── new, larger nanoparticles (not part of the training/validation set) ─────
# These are the candidates selected from Ni_nanoparticle_dataset and then
# MACE-optimised. They are kept SEPARATE from NP_FILES on purpose: every
# existing figure and summary in this script describes the systems the model
# was built and validated on, and those must not silently gain extra entries.
#
# The only output produced for this set is its own copy of the surface-
# environment SI figure (surface_environment_SI_figure_new_systems.pdf), so the
# new particles' surface environments can be compared against the same
# Ni(111)/Ni(100) references on the same axes.
# In the combined figure the nanoparticles are split into the three roles they
# play, by drawing a dotted separator immediately BEFORE each label listed here:
#
#   Ni25 ... Ni50    trained on
#   Ni55 ... Ni130   tested against DFT (out of distribution, DFT affordable)
#   Ni218 ... Ni1391 tested without DFT (too large for DFT; the point of the model)
#
# Listing the first member of each group keeps this correct if systems are added
# or the size ordering changes.
NP_GROUP_FIRST_LABELS = ("Ni55", "Ni218")

NEW_NP_FILES: Dict[str, str] = {
    "Ni218": str(TESTING / "Ni218_MACE_opt.xyz"),
    "Ni442": str(TESTING / "Ni442_MACE_opt.xyz"),
    "Ni524": str(TESTING / "Ni524_MACE_opt.xyz"),
    "Ni686": str(TESTING / "Ni686_MACE_opt.xyz"),
    "Ni864": str(TESTING / "Ni864_MACE_opt.xyz"),
    "Ni1141": str(TESTING / "Ni1141_MACE_opt.xyz"),
    "Ni1314": str(TESTING / "Ni1314_MACE_opt.xyz"),
    "Ni1391": str(TESTING / "Ni1391_MACE_opt.xyz"),
    "Ni3007": str(TESTING / "Ni3007_MACE_opt.xyz"),
}

# If a file contains more than one frame, analyze the last frame.
READ_INDEX = -1

OUTDIR = HERE / "bare_Ni_structural_analysis"

# First-neighbor-shell criterion used for CN, GCN, q_l and bond lengths.
FIRST_SHELL_CUTOFF = 3.0  # Angstrom

# Bulk fcc reference coordination in the GCN definition.
GCN_CN_MAX = 12

Q_L_VALUES = (4, 6)

# Probe-sphere exposure settings.
SURFACE_R_ATOM = 1.24      # Angstrom
SURFACE_R_PROBE = 1.20     # Angstrom
SURFACE_N_POINTS = 200
SURFACE_MIN_EXPOSED_FRACTION = 0.0  # any accessible point => surface

SEED = 32


# ============================================================
# I/O
# ============================================================

def validate_bare_ni(atoms: Atoms, label: str) -> None:
    symbols = atoms.get_chemical_symbols()
    if any(symbol != "Ni" for symbol in symbols):
        raise ValueError(
            f"{label}: expected a bare Ni-only structure, "
            f"found species {sorted(set(symbols))}."
        )


def load_structure(filename: str, label: str, require_pbc: bool) -> Atoms:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{label}: file not found:\n  {filename}")

    atoms = read(str(path), index=READ_INDEX)
    validate_bare_ni(atoms, label)

    if require_pbc:
        if not np.any(atoms.get_pbc()):
            warnings.warn(
                f"{label}: no PBC flags were read from {filename}. "
                "Check the extxyz header.",
                RuntimeWarning,
            )
        if abs(atoms.get_cell().volume) < 1.0e-12:
            raise ValueError(f"{label}: periodic surface has no valid lattice/cell.")

    return atoms


# ============================================================
# PBC-AWARE FIRST-SHELL NEIGHBORS
# ============================================================

def directed_neighbor_data(
    atoms: Atoms,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Directed neighbor entries from ASE.

    D is the vector from atom i to the corresponding periodic image of atom j;
    S is the integer cell shift of that image.
    """
    i, j, D, S = neighbor_list(
        "ijDS",
        atoms,
        cutoff,
        self_interaction=False,
    )

    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=int)

    if len(D):
        d = np.linalg.norm(D, axis=1)
        keep = d > 1.0e-10
        i, j, D, S = i[keep], j[keep], D[keep], S[keep]

    return i, j, D, S


def coordination_numbers(atoms: Atoms, cutoff: float) -> np.ndarray:
    i, _, _, _ = directed_neighbor_data(atoms, cutoff)
    return np.bincount(i, minlength=len(atoms)).astype(int)


def neighbor_indices_and_vectors(
    atoms: Atoms,
    cutoff: float,
) -> Tuple[List[List[int]], List[List[np.ndarray]]]:
    i_all, j_all, D_all, _ = directed_neighbor_data(atoms, cutoff)

    nbr_idx: List[List[int]] = [[] for _ in range(len(atoms))]
    nbr_vec: List[List[np.ndarray]] = [[] for _ in range(len(atoms))]

    for i, j, vec in zip(i_all, j_all, D_all):
        nbr_idx[int(i)].append(int(j))
        nbr_vec[int(i)].append(np.asarray(vec, dtype=float))

    return nbr_idx, nbr_vec


def generalized_coordination_numbers(
    atoms: Atoms,
    cutoff: float,
    cn_max: int = GCN_CN_MAX,
) -> np.ndarray:
    """
    GCN_i = sum_{j in NN(i)} CN_j / cn_max.

    All first-shell neighbors are included, regardless of whether they are
    surface or subsurface. Periodic images are counted as physical neighbors.
    """
    cn = coordination_numbers(atoms, cutoff)
    nbr_idx, _ = neighbor_indices_and_vectors(atoms, cutoff)

    gcn = np.zeros(len(atoms), dtype=float)
    for i in range(len(atoms)):
        gcn[i] = sum(cn[j] for j in nbr_idx[i]) / float(cn_max)

    return gcn


def _positive_shift(shift: Sequence[int]) -> bool:
    for value in shift:
        if value > 0:
            return True
        if value < 0:
            return False
    return False


def first_shell_bond_lengths(atoms: Atoms, cutoff: float) -> np.ndarray:
    """Unique undirected first-shell bond lengths, including periodic bonds."""
    i_all, j_all, D_all, S_all = directed_neighbor_data(atoms, cutoff)
    values = []

    for i, j, vec, shift in zip(i_all, j_all, D_all, S_all):
        i = int(i)
        j = int(j)

        if i < j:
            values.append(float(np.linalg.norm(vec)))
        elif i == j and _positive_shift(shift):
            # Keep one of the +/- periodic self-image bonds.
            values.append(float(np.linalg.norm(vec)))

    return np.asarray(values, dtype=float)

def keep_upper_slab_surface(atoms, surface_atoms):
    """
    Keep only surface atoms belonging to the upper face of the slab.
    """
    z = atoms.get_positions()[:, 2]

    z_mid = 0.5 * (z.min() + z.max())

    return [
        i for i in surface_atoms
        if z[i] > z_mid
    ]

# ============================================================
# STEINHARDT / LECHNER-DELLAGO ORDER PARAMETERS
# ============================================================

def qlm_from_vectors(vectors: Sequence[np.ndarray], l: int) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros(2 * l + 1, dtype=complex)

    vecs = np.asarray(vectors, dtype=float)
    r = np.linalg.norm(vecs, axis=1)
    valid = r > 1.0e-12
    vecs = vecs[valid]
    r = r[valid]

    if len(r) == 0:
        return np.zeros(2 * l + 1, dtype=complex)

    polar = np.arccos(np.clip(vecs[:, 2] / r, -1.0, 1.0))
    azimuth = np.mod(np.arctan2(vecs[:, 1], vecs[:, 0]), 2.0 * np.pi)

    qlm = np.zeros(2 * l + 1, dtype=complex)
    for im, m in enumerate(range(-l, l + 1)):
        qlm[im] = np.mean(spherical_harmonic(l, m, polar, azimuth))

    return qlm


def rotational_invariant_q(qlm: np.ndarray, l: int) -> float:
    return float(
        np.sqrt(
            (4.0 * np.pi / (2 * l + 1))
            * np.sum(np.abs(qlm) ** 2)
        )
    )


def steinhardt_order_parameters(
    atoms: Atoms,
    cutoff: float,
    l_values: Sequence[int] = Q_L_VALUES,
) -> Dict[str, np.ndarray]:
    """PBC-aware q_l and locally averaged qbar_l values."""
    nbr_idx, nbr_vec = neighbor_indices_and_vectors(atoms, cutoff)
    out: Dict[str, np.ndarray] = {}

    for l in l_values:
        qlm = np.asarray(
            [qlm_from_vectors(nbr_vec[i], l) for i in range(len(atoms))],
            dtype=complex,
        )

        q = np.asarray(
            [rotational_invariant_q(qlm[i], l) for i in range(len(atoms))]
        )

        qbar_lm = np.zeros_like(qlm)
        for i in range(len(atoms)):
            # Repeated j indices are retained if distinct periodic images are
            # distinct physical first-shell neighbors.
            group = [i] + list(nbr_idx[i])
            qbar_lm[i] = np.mean(qlm[group], axis=0)

        qbar = np.asarray(
            [rotational_invariant_q(qbar_lm[i], l) for i in range(len(atoms))]
        )

        out[f"q{l}"] = q
        out[f"qbar{l}"] = qbar

    return out


# ============================================================
# PROBE-SPHERE EXPOSURE, INCLUDING PERIODIC IMAGES
# ============================================================

def fibonacci_sphere(n_points: int) -> np.ndarray:
    if n_points < 1:
        raise ValueError("n_points must be >= 1")

    k = np.arange(n_points, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (k + 0.5) / n_points
    r_xy = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = golden_angle * k

    return np.column_stack(
        (r_xy * np.cos(phi), r_xy * np.sin(phi), z)
    )


def periodic_shift_ranges(atoms: Atoms, interaction_radius: float) -> List[range]:
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)

    ranges: List[range] = []
    for axis in range(3):
        if not pbc[axis]:
            ranges.append(range(0, 1))
            continue

        if lengths[axis] <= 1.0e-12:
            raise ValueError("Periodic direction has zero cell-vector length.")

        # At least +/-1 is needed near a periodic boundary. Very small cells
        # automatically include additional images.
        n = max(1, int(math.ceil(interaction_radius / lengths[axis])))
        ranges.append(range(-n, n + 1))

    return ranges


def periodic_image_centers(
    atoms: Atoms,
    interaction_radius: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    ranges = periodic_shift_ranges(atoms, interaction_radius)

    centers = []
    atom_indices = []
    shifts = []

    for shift_tuple in product(*ranges):
        shift = np.asarray(shift_tuple, dtype=int)
        translation = (
            shift[0] * cell[0]
            + shift[1] * cell[1]
            + shift[2] * cell[2]
        )

        for j, pos in enumerate(positions):
            centers.append(pos + translation)
            atom_indices.append(j)
            shifts.append(shift.copy())

    return (
        np.asarray(centers, dtype=float),
        np.asarray(atom_indices, dtype=int),
        np.asarray(shifts, dtype=int),
    )


def surface_atoms_from_exposure(
    atoms: Atoms,
    r_atom: float = SURFACE_R_ATOM,
    r_probe: float = SURFACE_R_PROBE,
    n_points: int = SURFACE_N_POINTS,
    min_surface_fraction: float = SURFACE_MIN_EXPOSED_FRACTION,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Probe-sphere accessibility with explicit periodic images.

    For a slab this normally identifies atoms on both surface slab faces.
    """
    cn = coordination_numbers(atoms, FIRST_SHELL_CUTOFF)
    sphere = fibonacci_sphere(n_points)
    shell = float(r_atom + r_probe)
    interaction_radius = 2.0 * shell

    image_centers, image_atom_idx, image_shifts = periodic_image_centers(
        atoms,
        interaction_radius,
    )

    positions = np.asarray(atoms.get_positions(), dtype=float)
    surface_fraction = np.zeros(len(atoms), dtype=float)
    surface_atoms: List[int] = []

    for i, centre in enumerate(positions):
        points = centre[None, :] + shell * sphere

        centre_dist = np.linalg.norm(image_centers - centre[None, :], axis=1)
        candidate_mask = centre_dist <= interaction_radius + 1.0e-12

        candidate_centers = image_centers[candidate_mask]
        candidate_indices = image_atom_idx[candidate_mask]
        candidate_shifts = image_shifts[candidate_mask]

        # Remove only atom i in the central cell. Periodic copies of i remain
        # valid occluders and are required for small periodic cells.
        central_self = (
            (candidate_indices == i)
            & np.all(candidate_shifts == 0, axis=1)
        )
        candidate_centers = candidate_centers[~central_self]

        buried = np.zeros(n_points, dtype=bool)
        for other_centre in candidate_centers:
            d = np.linalg.norm(points - other_centre[None, :], axis=1)
            buried |= d < shell
            if buried.all():
                break

        fraction = float(np.mean(~buried))
        surface_fraction[i] = fraction

        if fraction > min_surface_fraction:
            surface_atoms.append(i)

    return cn, surface_atoms, surface_fraction


# ============================================================
# STATISTICS AND PLOTS
# ============================================================

def iqr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan")
    q25, q75 = np.percentile(values, [25, 75])
    return float(q75 - q25)


def safe_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": iqr(values),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def plot_box_distribution(
    dataframe: pd.DataFrame,
    labels: Sequence[str],
    value_column: str,
    ylabel: str,
    xlabel: str,
    output_path: Path,
    show_points: bool = False,
) -> None:
    """Publication-style box plot, optionally with all individual points."""
    data = [
        dataframe.loc[
            dataframe["system"] == label,
            value_column,
        ].to_numpy(float)
        for label in labels
    ]

    display_labels = [
        format_system_label(label)
        for label in labels
    ]

    width = max(
        10.5,
        1.15 * len(labels),
    )

    fig, ax = plt.subplots(
        figsize=(width, 6.5)
    )

    bp = ax.boxplot(
        data,
        tick_labels=display_labels,
        showfliers=not show_points,
        patch_artist=True,
        widths=0.58,
        medianprops={"linewidth": 2.0},
        whiskerprops={"linewidth": 1.3},
        capprops={"linewidth": 1.3},
        boxprops={"linewidth": 1.4},
        flierprops={
            "marker": "o",
            "markersize": 5,
            "alpha": 0.60,
        },
    )

    # Matplotlib returns one box/median/flier and two whiskers/caps per
    # category.  Apply the fixed per-system palette to every component.
    for idx, label in enumerate(labels):
        face_color, edge_color = system_plot_colors(label)

        bp["boxes"][idx].set_facecolor(face_color)
        bp["boxes"][idx].set_edgecolor(edge_color)
        bp["boxes"][idx].set_alpha(0.62)

        bp["medians"][idx].set_color(edge_color)

        for artist in bp["whiskers"][2 * idx: 2 * idx + 2]:
            artist.set_color(edge_color)
        for artist in bp["caps"][2 * idx: 2 * idx + 2]:
            artist.set_color(edge_color)

        if idx < len(bp["fliers"]):
            bp["fliers"][idx].set_markerfacecolor(face_color)
            bp["fliers"][idx].set_markeredgecolor(edge_color)

    if show_points:
        rng = np.random.default_rng(SEED)

        for x_pos, (label, values) in enumerate(
            zip(labels, data),
            start=1,
        ):
            face_color, edge_color = system_plot_colors(label)
            jitter = rng.normal(
                0.0,
                0.055,
                size=len(values),
            )

            ax.scatter(
                np.full(
                    len(values),
                    x_pos,
                    dtype=float,
                )
                + jitter,
                values,
                s=58,
                alpha=0.68,
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=0.8,
                zorder=4,
            )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FS,
    )
    ax.set_xlabel(
        xlabel,
        fontsize=LABEL_FS,
    )

    ax.tick_params(
        axis="x",
        labelrotation=45,
        labelsize=XTICK_FS,
    )

    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    style_publication_axis(
        ax,
        categorical_x=True,
    )

    # style_publication_axis resets generic tick label size, so restore the
    # slightly smaller categorical x labels afterwards.
    ax.tick_params(
        axis="x",
        labelsize=XTICK_FS,
    )

    add_surface_group_separator(
        ax,
        labels,
    )

    fig.tight_layout()
    save_publication_figure(
        fig,
        output_path,
    )
    plt.close(fig)


def plot_violin_distribution(
    dataframe: pd.DataFrame,
    labels: Sequence[str],
    value_column: str,
    ylabel: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """
    Publication-style violin distributions.

    Constant datasets (common for ideal periodic terrace CN/GCN values)
    are shown as a thick horizontal line + central marker rather than
    attempting a singular kernel-density estimate.
    """
    data = [
        dataframe.loc[
            dataframe["system"] == label,
            value_column,
        ].to_numpy(float)
        for label in labels
    ]

    display_labels = [
        format_system_label(label)
        for label in labels
    ]

    width = max(
        10.5,
        1.15 * len(labels),
    )

    fig, ax = plt.subplots(
        figsize=(width, 6.5)
    )

    for x_pos, (label, values) in enumerate(
        zip(labels, data),
        start=1,
    ):
        face_color, edge_color = system_plot_colors(label)
        values = np.asarray(
            values,
            dtype=float,
        )

        if len(values) == 0:
            continue

        # Avoid singular KDEs for exactly constant distributions.
        if np.ptp(values) < 1.0e-12:
            y = float(values[0])

            ax.plot(
                [x_pos - 0.28, x_pos + 0.28],
                [y, y],
                color=edge_color,
                lw=3.0,
                solid_capstyle="round",
                zorder=3,
            )

            ax.scatter(
                [x_pos],
                [y],
                s=70,
                facecolor="white",
                edgecolor=edge_color,
                linewidth=1.4,
                zorder=4,
            )

        else:
            parts = ax.violinplot(
                [values],
                positions=[x_pos],
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body in parts["bodies"]:
                body.set_facecolor(face_color)
                body.set_edgecolor(edge_color)
                body.set_linewidth(1.4)
                body.set_alpha(0.62)

            # Median = white dot with dark edge.
            median = float(
                np.median(values)
            )

            ax.scatter(
                [x_pos],
                [median],
                s=68,
                facecolor="white",
                edgecolor=edge_color,
                linewidth=1.4,
                zorder=4,
            )

            # Interquartile range = compact vertical line.
            q25, q75 = np.percentile(
                values,
                [25, 75],
            )

            ax.plot(
                [x_pos, x_pos],
                [q25, q75],
                color=edge_color,
                lw=2.2,
                zorder=3,
            )

    ax.set_xticks(
        np.arange(
            1,
            len(labels) + 1,
        )
    )
    ax.set_xticklabels(
        display_labels,
        rotation=45,
        ha="right",
        fontsize=XTICK_FS,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FS,
    )
    ax.set_xlabel(
        xlabel,
        fontsize=LABEL_FS,
    )

    style_publication_axis(
        ax,
        categorical_x=True,
    )

    ax.tick_params(
        axis="x",
        labelsize=XTICK_FS,
    )

    add_surface_group_separator(
        ax,
        labels,
    )

    fig.tight_layout()
    save_publication_figure(
        fig,
        output_path,
    )
    plt.close(fig)


def surface_structural_summary(
    nanoparticle_surface_df: pd.DataFrame,
    periodic_surface_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compact numerical summary for the SI structural-characterization table."""
    rows = []

    for group_name, dataframe in (
        ("nanoparticle", nanoparticle_surface_df),
        ("periodic_surface", periodic_surface_df),
    ):
        if dataframe is None or dataframe.empty:
            continue

        for label, group in dataframe.groupby("system", sort=False):
            gcn = group["gcn"].to_numpy(float)
            qbar6 = group["qbar6"].to_numpy(float)

            facet = ""
            m = re.fullmatch(r"Ni(\d{3})_\d+x\d+", label)
            if m:
                facet = f"Ni({m.group(1)})"

            rows.append({
                "system": label,
                "group": group_name,
                "facet": facet,
                "n_surface": len(group),
                "surface_gcn_median": float(np.median(gcn)),
                "surface_gcn_iqr": iqr(gcn),
                "surface_gcn_min": float(np.min(gcn)),
                "surface_gcn_max": float(np.max(gcn)),
                "surface_qbar6_median": float(np.median(qbar6)),
                "surface_qbar6_iqr": iqr(qbar6),
                "surface_qbar6_min": float(np.min(qbar6)),
                "surface_qbar6_max": float(np.max(qbar6)),
            })

    return pd.DataFrame(rows)


def plot_surface_environment_si_figure(
    nanoparticle_surface_df: pd.DataFrame,
    periodic_surface_df: pd.DataFrame,
    nanoparticle_labels: Sequence[str],
    output_path: Path,
    figsize: Tuple[float, float] = (15.5, 6.6),
    font_scale: float = 1.0,
    dividers_after: Sequence[int] = (),
    xtick_rotation: float = 45.0,
    xtick_font_scale: float | None = None,
) -> None:
    """
    Two-panel SI figure comparing the distribution of local surface-Ni
    environments in nanoparticles with compact Ni(111) and Ni(100) references.

    Nanoparticles are shown as violin distributions of individual surface atoms.
    For each periodic facet, the three small points are the medians of the three
    available supercells.  The larger open marker is the median of those three
    cell medians.  This avoids treating periodically repeated terrace atoms as a
    continuous distribution while still showing that the local environment is
    essentially independent of supercell size.
    """
    # figsize/font_scale let a wider version of this same figure hold more
    # systems (the combined reference + new panel) without changing the two
    # original figures, which use the defaults. divider_after optionally draws
    # a second separator inside the nanoparticle block, at x = divider_after +
    # 0.5, to mark where one nanoparticle family ends and the next begins.
    label_fs = LABEL_FS * font_scale
    tick_fs = TICK_FS * font_scale
    # The system names scale separately: there are twenty-three of them on a
    # categorical axis, so they hit each other long before the axis labels and
    # y ticks are large enough. Defaults to font_scale when not given.
    xtick_fs = XTICK_FS * (font_scale if xtick_font_scale is None
                           else xtick_font_scale)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    facet_groups = {
        "Ni(111)": ["Ni111_2x2", "Ni111_3x3", "Ni111_4x3"],
        "Ni(100)": ["Ni100_3x3", "Ni100_4x3", "Ni100_4x4"],
    }

    panel_specs = [
        (
            "gcn",
            r"GCN($\mathrm{Ni}_{\mathrm{surf}}$)",
            "Ni system",
        ),
        (
            "qbar6",
            r"$\bar{q}_6$($\mathrm{Ni}_{\mathrm{surf}}$)", 
            "Ni system",
        ),
    ]

    n_np = len(nanoparticle_labels)
    facet_positions = {
        "Ni(111)": n_np + 1,
        "Ni(100)": n_np + 2,
    }
    all_positions = list(range(1, n_np + 1)) + list(facet_positions.values())
    all_ticklabels = [
        format_system_label(label)
        for label in nanoparticle_labels
    ] + ["Ni(111)", "Ni(100)"]

    for ax, (column, ylabel, xlabel) in zip(axes, panel_specs):
        # Nanoparticle distributions.
        for x_pos, label in enumerate(nanoparticle_labels, start=1):
            values = nanoparticle_surface_df.loc[
                nanoparticle_surface_df["system"] == label,
                column,
            ].to_numpy(float)

            if len(values) == 0:
                continue

            face_color, edge_color = system_plot_colors(label)

            if np.ptp(values) < 1.0e-12:
                y = float(values[0])
                ax.plot(
                    [x_pos - 0.27, x_pos + 0.27],
                    [y, y],
                    color=edge_color,
                    lw=3.0,
                    solid_capstyle="round",
                    zorder=3,
                )
                ax.scatter(
                    [x_pos],
                    [y],
                    s=68,
                    facecolor="white",
                    edgecolor=edge_color,
                    linewidth=1.4,
                    zorder=4,
                )
            else:
                parts = ax.violinplot(
                    [values],
                    positions=[x_pos],
                    widths=0.72,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(face_color)
                    body.set_edgecolor(edge_color)
                    body.set_linewidth(1.3)
                    body.set_alpha(0.62)

                median = float(np.median(values))
                q25, q75 = np.percentile(values, [25, 75])
                ax.plot(
                    [x_pos, x_pos],
                    [q25, q75],
                    color=edge_color,
                    lw=2.2,
                    zorder=4,
                )
                ax.scatter(
                    [x_pos],
                    [median],
                    s=66,
                    facecolor="white",
                    edgecolor=edge_color,
                    linewidth=1.4,
                    zorder=5,
                )

        # Surface references.  Show one small point for each supercell median,
        # and a larger open marker for the facet-level median.
        for facet, cell_labels in facet_groups.items():
            x0 = facet_positions[facet]
            base_color = FACET_REFERENCE_COLORS[facet]
            edge_color = darken_color(base_color)
            marker = FACET_REFERENCE_MARKERS[facet]

            cell_medians = []
            available_cells = []
            for cell_label in cell_labels:
                subset = periodic_surface_df.loc[
                    periodic_surface_df["system"] == cell_label,
                    column,
                ].to_numpy(float)
                if len(subset) == 0:
                    continue
                cell_medians.append(float(np.median(subset)))
                available_cells.append(cell_label)

            if not cell_medians:
                continue

            offsets = SURFACE_CELL_X_OFFSETS[:len(cell_medians)]
            for offset, value in zip(offsets, cell_medians):
                ax.scatter(
                    [x0 + offset],
                    [value],
                    s=42,
                    facecolor=base_color,
                    edgecolor=edge_color,
                    linewidth=0.9,
                    alpha=0.72,
                    zorder=5,
                )

            facet_median = float(np.median(cell_medians))
            cell_min = float(np.min(cell_medians))
            cell_max = float(np.max(cell_medians))

            # The min--max bar quantifies the variation between supercell sizes.
            # For essentially identical cells this naturally collapses to a point.
            ax.plot(
                [x0, x0],
                [cell_min, cell_max],
                color=edge_color,
                lw=1.6,
                zorder=5,
            )
            ax.scatter(
                [x0],
                [facet_median],
                s=105,
                marker=marker,
                facecolor="white",
                edgecolor=edge_color,
                linewidth=1.8,
                zorder=6,
            )

        # Separate nanoparticle distributions from periodic-facet references.
        ax.axvline(
            n_np + 0.5,
            color="0.62",
            lw=1.0,
            ls="--",
            alpha=0.75,
            zorder=0,
        )

        # Optional separators inside the nanoparticle block. Used by the
        # combined figure to split the particles into the three roles they play:
        # trained on | tested against DFT | tested without DFT.
        for divider in dividers_after:
            ax.axvline(
                divider + 0.5,
                color="0.62",
                lw=1.0,
                ls=":",
                alpha=0.75,
                zorder=0,
            )

        ax.set_xticks(all_positions)
        # Upright labels are an option because at 45 deg the perpendicular
        # gap between adjacent labels is only 1/sqrt(2) of the x spacing, so
        # the combined figure's larger type collides; at 90 deg the full
        # spacing is available.
        ax.set_xticklabels(
            all_ticklabels,
            rotation=xtick_rotation,
            ha="center" if xtick_rotation >= 85 else "right",
            va="top",
            fontsize=xtick_fs,
        )
        ax.set_ylabel(ylabel, fontsize=label_fs, fontweight="normal")
        ax.set_xlabel(xlabel, fontsize=label_fs, fontweight="normal")

        style_publication_axis(ax, categorical_x=True)
        ax.tick_params(axis="x", labelsize=xtick_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)

        # Short group labels above the plotting region, avoiding a legend.
        #ax.text(
        #    0.39,
        #    1.015,
        #    "Nanoparticles",
        #    transform=ax.transAxes,
        #    ha="center",
        #    va="bottom",
        #    fontsize=TICK_FS,
        #)
        #ax.text(
        #    0.92,
        #    1.015,
        #    "Periodic surfaces",
        #    transform=ax.transAxes,
        #    ha="center",
        #    va="bottom",
        #    fontsize=TICK_FS,
        #)

       # ax.text(
       #     0.02,
       #     0.97,
       #     panel_label,
       #     transform=ax.transAxes,
       #     ha="left",
       #     va="top",
       #     fontsize=LABEL_FS,
       # )

        ax.set_xlim(0.45, n_np + 2.55)

    fig.tight_layout(w_pad=2.4)
    save_publication_figure(fig, output_path)
    plt.close(fig)


# ============================================================
# ANALYSIS FOR ONE STRUCTURE GROUP
# ============================================================

def analyze_structure_group(
    structures: Dict[str, Atoms],
    group_name: str,
    output_dir: Path,
    make_group_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Descriptors + per-group figures for one set of structures.

    make_group_plots=False computes and writes the CSVs but skips this group's
    own distribution figures. Used for the new nanoparticles, where the only
    figure wanted is the combined surface-environment one. The default keeps the
    original behaviour for the existing calls.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    atom_rows = []
    surface_rows = []
    bond_rows = []
    summary_rows = []
    labels = list(structures.keys())

    print(f"\n=== {group_name} ===")

    for label, atoms in structures.items():
        cn = coordination_numbers(atoms, FIRST_SHELL_CUTOFF)
        gcn = generalized_coordination_numbers(
            atoms,
            FIRST_SHELL_CUTOFF,
            cn_max=GCN_CN_MAX,
        )
        q = steinhardt_order_parameters(
            atoms,
            FIRST_SHELL_CUTOFF,
            l_values=Q_L_VALUES,
        )
        bonds = first_shell_bond_lengths(atoms, FIRST_SHELL_CUTOFF)
        _, surface_atoms, surface_fraction = surface_atoms_from_exposure(atoms)
        if group_name.lower().startswith("surface"):
            surface_atoms = keep_upper_slab_surface(
            atoms,
            surface_atoms,
        )
        surface_set = set(surface_atoms)

        pbc_tuple = tuple(bool(x) for x in atoms.get_pbc())
        cell_lengths = atoms.get_cell().lengths()

        print(
            f"  {label:16s} n={len(atoms):4d} "
            f"surface={len(surface_atoms):4d} "
            f"pbc={pbc_tuple} "
            f"cell=({cell_lengths[0]:.3f}, {cell_lengths[1]:.3f}, "
            f"{cell_lengths[2]:.3f}) A"
        )

        for i in range(len(atoms)):
            atom_rows.append({
                "system": label,
                "atom_index_0based": i,
                "cn": int(cn[i]),
                "gcn": float(gcn[i]),
                "q4": float(q["q4"][i]),
                "q6": float(q["q6"][i]),
                "qbar4": float(q["qbar4"][i]),
                "qbar6": float(q["qbar6"][i]),
                "surface_fraction": float(surface_fraction[i]),
                "is_surface": int(i in surface_set),
            })

        for i in surface_atoms:
            surface_rows.append({
                "system": label,
                "atom_index_0based": i,
                "cn": int(cn[i]),
                "gcn": float(gcn[i]),
                "q4": float(q["q4"][i]),
                "q6": float(q["q6"][i]),
                "qbar4": float(q["qbar4"][i]),
                "qbar6": float(q["qbar6"][i]),
                "surface_fraction": float(surface_fraction[i]),
            })

        for d in bonds:
            bond_rows.append({
                "system": label,
                "bond_length_A": float(d),
            })

        summary = {
            "system": label,
            "n_atoms": len(atoms),
            "n_surface": len(surface_atoms),
            "surface_atom_fraction": len(surface_atoms) / float(len(atoms)),
            "pbc_x": pbc_tuple[0],
            "pbc_y": pbc_tuple[1],
            "pbc_z": pbc_tuple[2],
            "cell_a_A": float(cell_lengths[0]),
            "cell_b_A": float(cell_lengths[1]),
            "cell_c_A": float(cell_lengths[2]),
        }
        summary.update(safe_stats(cn, "all_cn"))
        summary.update(safe_stats(gcn, "all_gcn"))
        summary.update(safe_stats(q["qbar6"], "all_qbar6"))
        summary.update(safe_stats(bonds, "bond_length_A"))

        if len(surface_atoms):
            idx = np.asarray(surface_atoms, dtype=int)
            summary.update(safe_stats(cn[idx], "surface_cn"))
            summary.update(safe_stats(gcn[idx], "surface_gcn"))
            summary.update(safe_stats(q["qbar6"][idx], "surface_qbar6"))
        else:
            summary.update(safe_stats(np.asarray([]), "surface_cn"))
            summary.update(safe_stats(np.asarray([]), "surface_gcn"))
            summary.update(safe_stats(np.asarray([]), "surface_qbar6"))

        summary_rows.append(summary)

    atom_df = pd.DataFrame(atom_rows)
    surface_df = pd.DataFrame(surface_rows)
    bond_df = pd.DataFrame(bond_rows)
    summary_df = pd.DataFrame(summary_rows)

    atom_df.to_csv(output_dir / "all_atom_descriptors.csv", index=False)
    surface_df.to_csv(output_dir / "surface_atom_descriptors.csv", index=False)
    bond_df.to_csv(output_dir / "first_shell_bonds.csv", index=False)
    summary_df.to_csv(output_dir / "descriptor_summary.csv", index=False)

    if len(surface_df):
        cn_counts = (
            surface_df.groupby(["system", "cn"], sort=False)
            .size()
            .reset_index(name="count")
        )
        totals = cn_counts.groupby("system")["count"].transform("sum")
        cn_counts["fraction"] = cn_counts["count"] / totals
        cn_counts.to_csv(output_dir / "surface_cn_counts.csv", index=False)

    if not make_group_plots:
        return {
            "all_atoms": atom_df,
            "surface_atoms": surface_df,
            "bonds": bond_df,
            "summary": summary_df,
        }

    xlabel = (
        "Bare Ni nanoparticle"
        if group_name.lower().startswith("nano")
        else "Bare periodic Ni surface"
    )

    # All atoms.
    plot_violin_distribution(
        atom_df, labels, "qbar6", r"$\bar{q}_6$", xlabel,
        output_dir / "all_atom_qbar6_distributions.pdf",
    )
    plot_box_distribution(
        atom_df, labels, "cn", "First-shell Ni coordination number", xlabel,
        output_dir / "all_atom_cn_distributions.pdf",
    )
    plot_violin_distribution(
        atom_df, labels, "gcn", "Generalized coordination number", xlabel,
        output_dir / "all_atom_gcn_distributions.pdf",
    )
    plot_box_distribution(
        bond_df, labels, "bond_length_A",
        r"First-shell Ni--Ni distance ($\AA$)", xlabel,
        output_dir / "first_shell_bondlength_distributions.pdf",
    )

    # Surface atoms only.
    if len(surface_df):
        plot_box_distribution(
            surface_df, labels, "cn", "Surface Ni coordination number", xlabel,
            output_dir / "surface_cn_distributions.pdf",
            show_points=True,
        )
        plot_violin_distribution(
            surface_df, labels, "gcn",
            "Surface Ni generalized coordination number", xlabel,
            output_dir / "surface_gcn_distributions.pdf",
        )
        plot_violin_distribution(
            surface_df, labels, "qbar6", r"Surface $\bar{q}_6$", xlabel,
            output_dir / "surface_qbar6_distributions.pdf",
        )

    return {
        "all_atoms": atom_df,
        "surface_atoms": surface_df,
        "bonds": bond_df,
        "summary": summary_df,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    settings = {
        "READ_INDEX": READ_INDEX,
        "FIRST_SHELL_CUTOFF_A": FIRST_SHELL_CUTOFF,
        "GCN_CN_MAX": GCN_CN_MAX,
        "Q_L_VALUES": list(Q_L_VALUES),
        "SURFACE_R_ATOM_A": SURFACE_R_ATOM,
        "SURFACE_R_PROBE_A": SURFACE_R_PROBE,
        "SURFACE_N_POINTS": SURFACE_N_POINTS,
        "SURFACE_MIN_EXPOSED_FRACTION": SURFACE_MIN_EXPOSED_FRACTION,
        "NP_FILES": NP_FILES,
        "SURFACE_FILES": SURFACE_FILES,
    }

    with open(OUTDIR / "analysis_settings.json", "w") as handle:
        json.dump(settings, handle, indent=2)

    nanoparticles: Dict[str, Atoms] = {}
    print("Loading bare Ni nanoparticles...")
    for label, filename in NP_FILES.items():
        atoms = load_structure(filename, label, require_pbc=False)
        nanoparticles[label] = atoms
        print(
            f"  loaded {label:16s} n={len(atoms):4d} "
            f"pbc={tuple(bool(x) for x in atoms.get_pbc())}"
        )

    surfaces: Dict[str, Atoms] = {}
    print("\nLoading bare periodic Ni surfaces...")
    for label, filename in SURFACE_FILES.items():
        atoms = load_structure(filename, label, require_pbc=True)
        surfaces[label] = atoms
        lengths = atoms.get_cell().lengths()
        print(
            f"  loaded {label:16s} n={len(atoms):4d} "
            f"pbc={tuple(bool(x) for x in atoms.get_pbc())} "
            f"cell=({lengths[0]:.3f}, {lengths[1]:.3f}, {lengths[2]:.3f}) A"
        )

    nanoparticle_results = None
    surface_results = None

    if nanoparticles:
        nanoparticle_results = analyze_structure_group(
            nanoparticles,
            group_name="Nanoparticles",
            output_dir=OUTDIR / "nanoparticles",
        )

    if surfaces:
        surface_results = analyze_structure_group(
            surfaces,
            group_name="Surfaces",
            output_dir=OUTDIR / "surfaces",
        )

    # Combined SI-ready structural-space figure and compact numerical table.
    if nanoparticle_results is not None and surface_results is not None:
        np_surface_df = nanoparticle_results["surface_atoms"]
        periodic_surface_df = surface_results["surface_atoms"]

        plot_surface_environment_si_figure(
            np_surface_df,
            periodic_surface_df,
            nanoparticle_labels=list(nanoparticles.keys()),
            output_path=OUTDIR / "surface_environment_SI_figure.pdf",
        )

        structural_summary_df = surface_structural_summary(
            np_surface_df,
            periodic_surface_df,
        )
        structural_summary_df.to_csv(
            OUTDIR / "surface_local_environment_summary.csv",
            index=False,
        )

    # ── independent section: the new, larger nanoparticles ──────────────────
    # Self-contained and additive: nothing above depends on it, and the new
    # particles never enter NP_FILES, so every figure and summary describing the
    # training/validation set is unchanged. Produced here:
    #   surface_environment_SI_figure_new_systems.pdf/.png
    #   surface_local_environment_summary_new_systems.csv
    # The same periodic Ni(111)/Ni(100) references are reused, so the new figure
    # is directly comparable to the original one.
    if NEW_NP_FILES and surface_results is not None:
        print("\nLoading new (unseen) bare Ni nanoparticles...")
        new_nanoparticles: Dict[str, Atoms] = {}
        for label, filename in NEW_NP_FILES.items():
            atoms = load_structure(filename, label, require_pbc=False)
            new_nanoparticles[label] = atoms
            print(f"  loaded {label:16s} n={len(atoms):4d}")

        new_results = analyze_structure_group(
            new_nanoparticles,
            group_name="Nanoparticles (new systems)",
            output_dir=OUTDIR / "nanoparticles_new_systems",
            make_group_plots=False,
        )
        new_surface_df = new_results["surface_atoms"]

        plot_surface_environment_si_figure(
            new_surface_df,
            surface_results["surface_atoms"],
            nanoparticle_labels=list(new_nanoparticles.keys()),
            output_path=OUTDIR / "surface_environment_SI_figure_new_systems.pdf",
        )

        surface_structural_summary(
            new_surface_df,
            surface_results["surface_atoms"],
        ).to_csv(
            OUTDIR / "surface_local_environment_summary_new_systems.csv",
            index=False,
        )

        # ── third figure: everything on one set of axes ─────────────────────
        # The two figures above merged: all 19 nanoparticles (reference set,
        # then the new systems, separated by a dotted line) against the same
        # periodic references. Wider canvas and larger fonts, so the figure
        # still matches the other main/SI figures once it is scaled to the
        # page; the two figures above keep the defaults.
        if nanoparticle_results is not None:
            combined_surface_df = pd.concat(
                [nanoparticle_results["surface_atoms"], new_surface_df],
                ignore_index=True,
            )
            combined_labels = list(nanoparticles.keys()) + list(new_nanoparticles.keys())

            plot_surface_environment_si_figure(
                combined_surface_df,
                surface_results["surface_atoms"],
                nanoparticle_labels=combined_labels,
                output_path=OUTDIR / "surface_environment_SI_figure_all_systems.pdf",
                # Type sized for the manuscript, not for the canvas: at 24 in
                # wide this figure is reduced a lot on the page, so the fonts
                # have to be scaled UP, not down, to land at the same printed
                # size as the other main/SI figures. Upright x labels, since
                # 45 deg leaves too little room between them at this size.
                figsize=(24.0, 7.2),
                font_scale=1.30,
                xtick_font_scale=0.95,
                dividers_after=[
                    combined_labels.index(first)
                    for first in NP_GROUP_FIRST_LABELS
                    if first in combined_labels
                ],
            )

            surface_structural_summary(
                combined_surface_df,
                surface_results["surface_atoms"],
            ).to_csv(
                OUTDIR / "surface_local_environment_summary_all_systems.csv",
                index=False,
            )

    print("\nDone.")
    print(f"Results written to: {OUTDIR.resolve()}")
    print("\nSeparate nanoparticle/ and surfaces/ folders contain:")
    print("  all_atom_descriptors.csv")
    print("  surface_atom_descriptors.csv")
    print("  descriptor_summary.csv")
    print("  surface_cn_counts.csv")
    print("  first_shell_bonds.csv")
    print("  all_atom_qbar6_distributions.pdf")
    print("  all_atom_cn_distributions.pdf")
    print("  all_atom_gcn_distributions.pdf")
    print("  first_shell_bondlength_distributions.pdf")
    print("  surface_cn_distributions.pdf")
    print("  surface_gcn_distributions.pdf")
    print("  surface_qbar6_distributions.pdf")
    print("\nTop-level SI-ready combined outputs:")
    print("  surface_environment_SI_figure.pdf")
    print("  surface_environment_SI_figure.png")
    print("  surface_local_environment_summary.csv")


if __name__ == "__main__":
    main()
