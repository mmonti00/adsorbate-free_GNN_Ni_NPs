#!/usr/bin/env python3
"""Screen OH*/H* co-adsorption sites on bare Ni nanoparticles and slabs.

Workflow:
    1. Identify accessible Ni top (T), bridge (B), threefold hollow (H), and fourfold hollow (F) sites.
    2. Generate geometrically admissible O/H site pairs from the bare Ni structure.
    3. Predict H2O(g)-referenced dissociative adsorption energies with the energy GNN.
    4. Optionally predict exact-site retention with the binary NP/slab classifier.
    5. Save all scored candidates, ranked by predicted adsorption energy, to CSV.

Set ``SYSTEM_NAME`` and the screening options in USER SETTINGS below, then run:
    python training_scripts/prediction.py

The default paths follow the repository layout:

    models_checkpoints/
        M6_energy_model.pt
        retention_model.pt
        M6_energy_model_all_train_data.extxyz
        M6_energy_model_all_valid_data.extxyz
    ML_relaxed_bare_geom/MACE/testing/
        Ni91_MACE_opt.xyz

Paths are resolved relative to this file, so the script can be launched from
any working directory. The supplied train/validation structures are used only
for motif-support and geometric-OOD diagnostics.

The NP/slab domain is determined from the structure's PBC flags, not from its
filename or extension. Nanoparticle inputs must therefore have
``pbc="F F F"``. If any PBC component is true, the structure is treated as a
periodic slab and the slab graph construction is used.

Requires NumPy, PyTorch, PyTorch Geometric, ASE, and the matching model-training
Python modules and checkpoints. The training/validation extxyz files are read
only to compute diagnostic columns (motif support and geometric OOD).
They never affect model predictions, retention scores, or ranking. By default,
both files must be available to calculate these diagnostics.

Note: these predictions are for candidate adsorption-site pairs on a bare Ni
geometry; they do not replace structural relaxation or DFT validation.
"""

import csv
import hashlib
import importlib
import inspect
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ase.io import read
from ase.geometry import find_mic
from torch_geometric.data import Data

# Import the architecture and graph conventions associated with MODEL_PATH.
ENERGY_TRAIN_MODULE = "train_M6_energy_model"
energy_module = importlib.import_module(ENERGY_TRAIN_MODULE)

BareSiteGCNN = energy_module.BareSiteGCNN
DEVICE = energy_module.DEVICE
EDGE_CUTOFF = energy_module.EDGE_CUTOFF
LOCAL_CUTOFF = energy_module.LOCAL_CUTOFF
# Use the Ni-O anchor definitions saved with the energy-model training code.
NI_O_CUTOFF = energy_module.NI_O_CUTOFF
NI_O_FOLDED_CUTOFF = energy_module.NI_O_FOLDED_CUTOFF
O_FOLDED_N_SHORT = energy_module.O_FOLDED_N_SHORT
O_FOLDED_N_TOTAL = energy_module.O_FOLDED_N_TOTAL
MAX_STANDARD_O_ANCHORS = energy_module.MAX_STANDARD_O_ANCHORS

# Use the corresponding Ni-H anchor definitions.
NI_H_CUTOFF = energy_module.NI_H_CUTOFF
NI_H_FOLDED_CUTOFF = energy_module.NI_H_FOLDED_CUTOFF
H_FOLDED_N_SHORT = energy_module.H_FOLDED_N_SHORT
H_FOLDED_N_TOTAL = energy_module.H_FOLDED_N_TOTAL
MAX_STANDARD_H_ANCHORS = energy_module.MAX_STANDARD_H_ANCHORS

classify_site = energy_module.classify_site

# Domain-specific graph cutoffs, defined by the energy training script.
LOCAL_CUTOFF_BY_DOMAIN = energy_module.LOCAL_CUTOFF_BY_DOMAIN
EDGE_CUTOFF_BY_DOMAIN = energy_module.EDGE_CUTOFF_BY_DOMAIN


def _get_cutoff(by_domain: Dict[str, object], domain_label: str, fallback: float) -> float:
    v = by_domain.get(domain_label) if by_domain else None
    return float(v) if v is not None else float(fallback)


def get_local_cutoff(domain_label: str) -> float:
    """LOCAL_CUTOFF to use for a structure of the given domain ('NP' or 'SLAB')."""
    return _get_cutoff(LOCAL_CUTOFF_BY_DOMAIN, domain_label, LOCAL_CUTOFF)


def get_edge_cutoff(domain_label: str) -> float:
    """EDGE_CUTOFF to use for a structure of the given domain ('NP' or 'SLAB')."""
    return _get_cutoff(EDGE_CUTOFF_BY_DOMAIN, domain_label, EDGE_CUTOFF)


def domain_label_for(atoms) -> str:
    """Classify the domain from PBC: any periodic direction means ``SLAB``."""
    try:
        pbc = atoms.get_pbc()
    except AttributeError:
        return "NP"
    return "SLAB" if any(bool(p) for p in pbc) else "NP"

# Coordination numbers for the optional CN-based surface detector.
compute_coordination_numbers = energy_module.compute_coordination_numbers

# ============================================================
# USER SETTINGS — input, models, and screening options
# ============================================================

# Repository paths. Change these only if the repository layout is modified.
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
CHECKPOINT_DIR = REPOSITORY_ROOT / "models_checkpoints"
TEST_GEOMETRY_DIR = REPOSITORY_ROOT / "ML_relaxed_bare_geom" / "MACE" / "testing"

# System to screen. The default input is the MACE-relaxed bare Ni91 particle.
SYSTEM_NAME = "Ni91"
MODEL_PATH = CHECKPOINT_DIR / "M6_energy_model.pt"
BARE_CLUSTER_FILE = TEST_GEOMETRY_DIR / f"{SYSTEM_NAME}_MACE_opt.xyz"
if not BARE_CLUSTER_FILE.is_file():
    # Periodic test structures are stored as extxyz so their cell and PBC flags
    # travel with the coordinates.
    BARE_CLUSTER_FILE = TEST_GEOMETRY_DIR / f"{SYSTEM_NAME}_MACE_opt.extxyz"

RANKING_FILENAME = {
    "Ni111_4x3": "Ni111_ranked_candidates.csv",
    "Ni100_4x4": "Ni100_ranked_candidates.csv",
}.get(SYSTEM_NAME, f"{SYSTEM_NAME}_ranked_candidates.csv")
# New rankings are written outside test_systems/, whose rankings are the
# reference data used in the paper; an existing file there is never overwritten.
OUT_CSV_ALL = (
    REPOSITORY_ROOT
    / "predictions"
    / SYSTEM_NAME
    / RANKING_FILENAME
)
REFERENCE_RANKING_DIR = REPOSITORY_ROOT / "test_systems"

# Used ONLY for motif-support/OOD columns, not for GNN prediction. These are
# stored beside the released energy checkpoint.
TRAIN_SUPPORT_EXTXYZ = CHECKPOINT_DIR / "M6_energy_model_all_train_data.extxyz"
VAL_SUPPORT_EXTXYZ = CHECKPOINT_DIR / "M6_energy_model_all_valid_data.extxyz"

SUPPORT_PERCENTILES = (1, 99)

# The optional classifier estimates whether *both* O/H anchor sets and the
# adsorption motif survive relaxation (not just whether the motif is unchanged).
USE_RETENTION_MODEL = True
RETENTION_MODEL_PATH = CHECKPOINT_DIR / "retention_model.pt"
# Import the classifier training module matching RETENTION_MODEL_PATH.
RETENTION_TRAIN_MODULE = "train_retention_classifier"

# Interpretation thresholds for the retention score. The score is continuous;
# none of these thresholds affects predicted energies or the energy ranking.
RETENTION_SCREENING_THRESHOLD = 0.50
RETENTION_CONSERVATIVE_THRESHOLD = 0.80
RETENTION_HIGHCONF_THRESHOLD = 0.95

TOP_N_PRINT = 30
PRINT_SURFACE_DETAILS = False  # Set True for per-atom indices and CN in the log.

# Ni-Ni graph used for surface detection / site generation
NI_NEIGHBOR_CUTOFF = 3.0
SURFACE_CN_THRESHOLD = 10  # Surface CN threshold when SURFACE_METHOD == "cn".

# Reject periodic loops that do not close within one local geometric ring.
RING_CLOSURE_TOL = 0.5  # Å

# Surface detector: "exposure" (probe accessibility) or "cn" (coordination).
SURFACE_METHOD = "exposure"

# Probe-based surface detection settings (used only with "exposure").
SURFACE_R_ATOM = 1.24                 # Ni radius (Å): metallic ~1.25, covalent ~1.24
SURFACE_R_PROBE = 1.2                 # adsorbate-sized probe (Å)
SURFACE_N_POINTS = 200                # test points per atom shell
SURFACE_MIN_EXPOSED_FRACTION = 0.00   # >this fraction of accessible points => surface

# Candidate generation toggles
INCLUDE_TOP = True
INCLUDE_BRIDGE = True
INCLUDE_HOLLOW = True
INCLUDE_HOLLOW4 = True          # Fourfold square hollow (FCC 100-type motif)
INCLUDE_SQUARE_DERIVED_HOLLOW = True # also score the 4 triangular subsets of each F square

# Geometric filtering between O-site and H-site candidates. Set these values
# for the system being screened; they are intentionally not selected
# automatically from the NP/slab domain.
MIN_SITE_DIST = 1.8   # Angstrom; use the desired value for the current system.
MAX_SITE_DIST = 6.5   # Angstrom

# Optional cap on number of candidates to score (None = all)
MAX_CANDIDATES = None

# O/H-site separation is measured between site centroids, using MIC if periodic.
# Pairs beyond PAIR_DISTANCE_SPLIT are designated "far".
PAIR_DISTANCE_SPLIT = 5.0

# If True, skip far pairs *before* building graphs or calling either GNN.
# The CSV then contains near pairs only; candidate IDs depend on this option.
SCORE_NEAR_ONLY = True

# A bare anchor-index tuple cannot identify which side of the cell is intended
# when an intra-site anchor pair, or the O/H site separation, lies at an exact
# half-cell tie. Training resolves those cases from the actual adsorbate
# coordinates; bare inference has no equivalent information. An O/H tie is
# flagged only when the tied images give different graphs (symmetric slabs give
# identical mirror-image graphs). Skip flagged candidates by default instead of
# scoring an arbitrary periodic image.
SKIP_PBC_AMBIGUOUS = True

# Slabs are detected from the PBC flags in the input structure. USE_PBC
# enables MIC-aware site finding; slab GNN graphs use explicit periodic images.
USE_PBC = True

# Limit slab candidates to the exposed top side when requested.
TOP_SURFACE_ONLY = True
TOP_AXIS = 2

# The bare input structure is set in main() and used by geometry helpers.
CURRENT_ATOMS = None

# RBF parameters must match the energy-model training module.
RBF_N_CENTERS = energy_module.N_RBF
RBF_R_MIN = energy_module.RBF_DMIN
RBF_R_MAX = energy_module.RBF_DMAX
RBF_GAMMA = energy_module.RBF_GAMMA


# ============================================================
# HELPERS
# ============================================================

def validate_user_settings() -> None:
    """Validate user-editable paths and scalar options before inference."""
    required_paths = {
        "energy checkpoint": MODEL_PATH,
        "bare Ni geometry": BARE_CLUSTER_FILE,
        "training support split": TRAIN_SUPPORT_EXTXYZ,
        "validation support split": VAL_SUPPORT_EXTXYZ,
    }
    if USE_RETENTION_MODEL:
        required_paths["retention checkpoint"] = RETENTION_MODEL_PATH

    missing = [
        f"{label}: {Path(path)}"
        for label, path in required_paths.items()
        if path is None or not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required input files were not found:\n  " + "\n  ".join(missing)
        )

    if not str(SYSTEM_NAME).strip():
        raise ValueError("SYSTEM_NAME must not be empty.")
    if SURFACE_METHOD not in {"exposure", "cn"}:
        raise ValueError("SURFACE_METHOD must be either 'exposure' or 'cn'.")
    if not (0 <= TOP_AXIS <= 2):
        raise ValueError("TOP_AXIS must be 0, 1, or 2.")
    if NI_NEIGHBOR_CUTOFF <= 0:
        raise ValueError("NI_NEIGHBOR_CUTOFF must be positive.")
    if MIN_SITE_DIST < 0 or MAX_SITE_DIST <= MIN_SITE_DIST:
        raise ValueError(
            "Require 0 <= MIN_SITE_DIST < MAX_SITE_DIST for candidate generation."
        )
    if PAIR_DISTANCE_SPLIT <= 0:
        raise ValueError("PAIR_DISTANCE_SPLIT must be positive.")
    if MAX_CANDIDATES is not None and MAX_CANDIDATES <= 0:
        raise ValueError("MAX_CANDIDATES must be positive or None.")
    if not (
        0.0 <= RETENTION_SCREENING_THRESHOLD
        <= RETENTION_CONSERVATIVE_THRESHOLD
        <= RETENTION_HIGHCONF_THRESHOLD
        <= 1.0
    ):
        raise ValueError(
            "Retention thresholds must be ordered between zero and one."
        )
    if len(SUPPORT_PERCENTILES) != 2 or not (
        0.0 <= SUPPORT_PERCENTILES[0] < SUPPORT_PERCENTILES[1] <= 100.0
    ):
        raise ValueError(
            "SUPPORT_PERCENTILES must contain two increasing values in [0, 100]."
        )
    output = Path(OUT_CSV_ALL).resolve()
    if output.is_relative_to(REFERENCE_RANKING_DIR.resolve()) and output.exists():
        raise FileExistsError(
            f"{output} is a reference ranking in test_systems/ and would be "
            "overwritten; set OUT_CSV_ALL to a new path."
        )


def report_resolved_paths() -> None:
    """Print the files used for this inference run."""
    print("=== INPUTS ===")
    print(f"System:                    {SYSTEM_NAME}")
    print(f"Bare Ni geometry:          {BARE_CLUSTER_FILE}")
    print(f"Energy checkpoint:         {MODEL_PATH}")
    if USE_RETENTION_MODEL:
        print(f"Retention checkpoint:      {RETENTION_MODEL_PATH}")
    else:
        print("Retention checkpoint:      disabled")
    print(f"Training support split:    {TRAIN_SUPPORT_EXTXYZ}")
    print(f"Validation support split:  {VAL_SUPPORT_EXTXYZ}")
    print(f"Output ranking:            {OUT_CSV_ALL}")


def repository_relative(path: Path) -> str:
    """Return a portable repository-relative path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_prediction_settings(
    output_csv: Path,
    domain_label: str,
    n_candidates: int,
    energy_checkpoint: dict,
    retention_checkpoint: Optional[dict],
) -> Path:
    """Write the complete user-facing inference settings beside the CSV."""
    settings_path = output_csv.with_name(f"{output_csv.stem}_settings.json")
    payload = {
        "schema_version": 1,
        "system_name": SYSTEM_NAME,
        "domain": domain_label,
        "bare_structure": repository_relative(BARE_CLUSTER_FILE),
        "output_csv": repository_relative(output_csv),
        "n_candidates_written": int(n_candidates),
        "models": {
            "energy_checkpoint": repository_relative(MODEL_PATH),
            "energy_checkpoint_sha256": file_sha256(MODEL_PATH),
            "energy_training_module": ENERGY_TRAIN_MODULE,
            "retention_enabled": bool(USE_RETENTION_MODEL),
            "retention_checkpoint": (
                repository_relative(RETENTION_MODEL_PATH)
                if USE_RETENTION_MODEL
                else None
            ),
            "retention_checkpoint_sha256": (
                file_sha256(RETENTION_MODEL_PATH)
                if USE_RETENTION_MODEL
                else None
            ),
            "retention_training_module": (
                RETENTION_TRAIN_MODULE if USE_RETENTION_MODEL else None
            ),
        },
        "graph": {
            "local_cutoff_A": float(get_local_cutoff(domain_label)),
            "edge_cutoff_A": float(get_edge_cutoff(domain_label)),
            "use_pbc": bool(USE_PBC),
        },
        "surface_detection": {
            "method": SURFACE_METHOD,
            "ni_neighbor_cutoff_A": float(NI_NEIGHBOR_CUTOFF),
            "surface_cn_threshold": int(SURFACE_CN_THRESHOLD),
            "ring_closure_tolerance_A": float(RING_CLOSURE_TOL),
            "probe_atom_radius_A": float(SURFACE_R_ATOM),
            "probe_radius_A": float(SURFACE_R_PROBE),
            "probe_points_per_atom": int(SURFACE_N_POINTS),
            "minimum_exposed_fraction": float(SURFACE_MIN_EXPOSED_FRACTION),
            "top_surface_only": bool(TOP_SURFACE_ONLY),
            "top_axis": int(TOP_AXIS),
        },
        "candidate_generation": {
            "include_top": bool(INCLUDE_TOP),
            "include_bridge": bool(INCLUDE_BRIDGE),
            "include_hollow": bool(INCLUDE_HOLLOW),
            "include_hollow4": bool(INCLUDE_HOLLOW4),
            "include_square_derived_hollow": bool(
                INCLUDE_SQUARE_DERIVED_HOLLOW
            ),
            "minimum_site_distance_A": float(MIN_SITE_DIST),
            "maximum_site_distance_A": float(MAX_SITE_DIST),
            "pair_distance_split_A": float(PAIR_DISTANCE_SPLIT),
            "score_near_only": bool(SCORE_NEAR_ONLY),
            "skip_pbc_ambiguous": bool(SKIP_PBC_AMBIGUOUS),
            "maximum_candidates": (
                None if MAX_CANDIDATES is None else int(MAX_CANDIDATES)
            ),
        },
        "retention_thresholds": {
            "screening": float(RETENTION_SCREENING_THRESHOLD),
            "conservative": float(RETENTION_CONSERVATIVE_THRESHOLD),
            "high_confidence": float(RETENTION_HIGHCONF_THRESHOLD),
        },
        "training_support": {
            "training_split": repository_relative(TRAIN_SUPPORT_EXTXYZ),
            "validation_split": repository_relative(VAL_SUPPORT_EXTXYZ),
            "support_percentiles": [float(v) for v in SUPPORT_PERCENTILES],
        },
        "checkpoint_training_settings": {
            "energy": energy_checkpoint.get("settings", {}),
            "retention": (
                retention_checkpoint.get("settings", {})
                if retention_checkpoint is not None
                else None
            ),
        },
    }
    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return settings_path


def validate_hybrid_checkpoint_settings(ckpt: dict) -> None:
    """Fail early if MODEL_PATH was not trained with these hybrid O/H rules."""
    settings = ckpt.get("settings", {})
    expected = {
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

    missing = [key for key in expected if key not in settings]
    if missing:
        raise RuntimeError(
            "The energy checkpoint does not contain the hybrid O/H settings "
            f"{missing}. Update MODEL_PATH to the checkpoint produced by "
            f"{ENERGY_TRAIN_MODULE}.py."
        )

    mismatches = []
    for key, expected_value in expected.items():
        actual_value = settings[key]
        if float(actual_value) != float(expected_value):
            mismatches.append(
                f"{key}: checkpoint={actual_value}, module={expected_value}"
            )

    if mismatches:
        raise RuntimeError(
            "Energy checkpoint and imported hybrid training module disagree:\n  "
            + "\n  ".join(mismatches)
        )

    expected_definitions = {
        "O_ANCHOR_DEFINITION": "hybrid_2p20_standard_at_least_2short_four_total_F",
        "H_ANCHOR_DEFINITION": "hybrid_2p00_standard_at_least_2short_four_total_F",
    }
    definition_mismatches = []
    for key, expected_value in expected_definitions.items():
        actual_value = settings.get(key)
        if actual_value != expected_value:
            definition_mismatches.append(
                f"{key}: checkpoint={actual_value!r}, expected={expected_value!r}"
            )

    if definition_mismatches:
        raise RuntimeError(
            "Energy checkpoint uses a different anchor-rule definition:\n  "
            + "\n  ".join(definition_mismatches)
        )

    expected_graph_settings = {
        "LOCAL_CUTOFF": LOCAL_CUTOFF,
        "EDGE_CUTOFF": EDGE_CUTOFF,
        "N_RBF": RBF_N_CENTERS,
        "RBF_DMIN": RBF_R_MIN,
        "RBF_DMAX": RBF_R_MAX,
        "RBF_GAMMA": RBF_GAMMA,
    }
    graph_mismatches = []
    for key, expected_value in expected_graph_settings.items():
        actual_value = settings.get(key)
        if actual_value is None or not np.isclose(
            float(actual_value),
            float(expected_value),
            rtol=0.0,
            atol=1e-12,
        ):
            graph_mismatches.append(
                f"{key}: checkpoint={actual_value}, module={expected_value}"
            )

    if graph_mismatches:
        raise RuntimeError(
            "Energy checkpoint and imported module use different graph settings:\n  "
            + "\n  ".join(graph_mismatches)
        )


def load_model(model_path: str):
    ckpt = torch.load(model_path, map_location=DEVICE)
    validate_hybrid_checkpoint_settings(ckpt)

    if not bool(ckpt.get("energy_only", False)):
        raise RuntimeError(
            "MODEL_PATH does not point to an energy-only checkpoint."
        )

    settings = ckpt.get("settings", {})
    if settings.get("TRAINING_OBJECTIVE") != "energy_only":
        raise RuntimeError(
            "Checkpoint TRAINING_OBJECTIVE is not 'energy_only'."
        )
    if settings.get("TARGET_REFERENCE") != "H2O_gas":
        raise RuntimeError(
            "Checkpoint TARGET_REFERENCE is not 'H2O_gas'."
        )

    model = BareSiteGCNN(
        node_dim=ckpt["node_dim"],
        edge_dim=ckpt["edge_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_retention_model(model_path: str, module_name: str):
    """Load the binary exact-site retention GNN with separate NP/slab heads."""
    cls_module = importlib.import_module(module_name)
    ckpt = torch.load(model_path, map_location=DEVICE)
    expected_task = "retained_exact_binary_two_domain_heads"
    if ckpt.get("task") != expected_task:
        raise ValueError(
            f"Unsupported retention checkpoint task {ckpt.get('task')!r}; "
            f"expected {expected_task!r}."
        )

    domain_to_id = getattr(cls_module, "DOMAIN_TO_ID", None)
    if domain_to_id != ckpt.get("domain_to_id"):
        raise ValueError(
            "The retention checkpoint and training module use different "
            "NP/SLAB domain mappings."
        )

    model = cls_module.MotifRetentionTwoHeadGNN(
        node_dim=ckpt["node_dim"],
        edge_dim=ckpt["edge_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, cls_module


def predict_retention_probability(retention_model, data: Data, domain_label: str, cls_module) -> float:
    """Predict p(retained_exact) from the relevant NP or slab binary head."""
    data = data.to(DEVICE)
    batch_vec = torch.zeros(data.x.shape[0], dtype=torch.long, device=DEVICE)
    domain_id = torch.tensor(
        [cls_module.DOMAIN_TO_ID[domain_label]], dtype=torch.long, device=DEVICE
    )
    with torch.no_grad():
        logits = retention_model(
            data.x, data.edge_index, data.edge_attr, batch_vec, domain_id
        )
        return float(torch.sigmoid(logits).item())


def retention_flag_from_probability(p_retained_exact: float) -> str:
    """Human-readable exact-site retention/migration flag.

    The probability is p(retained_exact), where retained_exact means:
    same motif + same O-anchor atoms + same H-anchor atoms.
    Anything else is treated as migration for ranking purposes.
    """
    if np.isnan(p_retained_exact):
        return "not_evaluated"
    if p_retained_exact >= RETENTION_HIGHCONF_THRESHOLD:
        return "likely_retained"
    if p_retained_exact >= RETENTION_CONSERVATIVE_THRESHOLD:
        return "possible_retained"
    if p_retained_exact >= RETENTION_SCREENING_THRESHOLD:
        return "uncertain"
    if p_retained_exact < 0.10:
        return "very_likely_migrates"
    return "likely_migrates"


def fmt_score(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x):.6f}"


def classify_site_extended(n: int) -> str:
    """Convert an anchor count into a T/B/H/F site label."""
    if n == 4:
        return "F"   # Four-fold / square hollow
    return classify_site(n)


def motif_from_anchor_counts(n_o: int, n_h: int) -> str:
    return classify_site_extended(n_o) + classify_site_extended(n_h)


def site_type_from_anchors(site: Tuple[int, ...]) -> str:
    n = len(site)
    if n <= 1:
        return "T"
    if n == 2:
        return "B"
    if n == 3:
        return "H"
    return "F"   # four-fold square hollow


def candidate_is_allowed(
    o_site: Tuple[int, ...],
    h_site: Tuple[int, ...],
    positions: np.ndarray,
) -> bool:
    """
    Motif-aware exclusion rules + geometric filtering.
    """
    o_set = set(o_site)
    h_set = set(h_site)

    o_type = site_type_from_anchors(o_site)
    h_type = site_type_from_anchors(h_site)

    # identical anchor sets are never useful
    if o_set == h_set:
        return False

    # forbid top on an atom already used by bridge/hollow/square of the other adsorbate
    if o_type == "T" and h_type in {"B", "H", "F"} and o_set.issubset(h_set):
        return False
    if h_type == "T" and o_type in {"B", "H", "F"} and h_set.issubset(o_set):
        return False

    # forbid bridge that is one edge of the other hollow/square
    if o_type == "B" and h_type in {"H", "F"} and o_set.issubset(h_set):
        return False
    if h_type == "B" and o_type in {"H", "F"} and h_set.issubset(o_set):
        return False

    # forbid 3-hollow that is a face of the other square hollow
    if o_type == "H" and h_type == "F" and o_set.issubset(h_set):
        return False
    if h_type == "H" and o_type == "F" and h_set.issubset(o_set):
        return False

    # geometric consistency filter
    if sites_too_close(o_site, h_site, positions):
        return False

    return True


def generate_top_sites(surface_atoms: List[int]) -> List[Tuple[int, ...]]:
    return [(i,) for i in surface_atoms]


def generate_bridge_sites(surface_atoms: List[int], neighbor_map: Dict[int, set]) -> List[Tuple[int, ...]]:
    sites = set()
    for i in surface_atoms:
        for j in neighbor_map[i]:
            if i < j:
                sites.add((i, j))
    return sorted(sites)


def ring_closes_locally(cycle: List[int], positions: np.ndarray) -> bool:
    """True if walking the ring bond by bond (each step by its MIC vector) comes
    back to the start, i.e. the ring is a real local motif and not a periodic
    row that only closes by wrapping once around the cell. Always True without PBC."""
    total = np.zeros(3, dtype=float)
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        total += mic_vector(np.asarray(positions[b], dtype=float) - np.asarray(positions[a], dtype=float))
    return float(np.linalg.norm(total)) < RING_CLOSURE_TOL


def generate_hollow_sites(
    surface_atoms: List[int],
    neighbor_map: Dict[int, set],
    positions: np.ndarray,
) -> List[Tuple[int, ...]]:
    """
    Hollow = compact triplet where all three atoms are pairwise neighbors,
    forming a triangle that closes locally (not a periodic row of 3).
    """
    sites = set()
    for i, j, k in combinations(surface_atoms, 3):
        if (j in neighbor_map[i]) and (k in neighbor_map[i]) and (k in neighbor_map[j]):
            if not ring_closes_locally([i, j, k], positions):
                continue
            sites.add(tuple(sorted((i, j, k))))
    return sorted(sites)


def generate_square_hollow_sites(
    surface_atoms: List[int],
    neighbor_map: Dict[int, set],
    positions: np.ndarray,
) -> List[Tuple[int, ...]]:
    """Find locally closed, uncapped fourfold rings on the exposed surface.

    Require four C4 bonds, local (not cell-spanning) ring closure, and no
    additional surface atom shared by all four corners.
    """
    sites = set()

    for quad in combinations(surface_atoms, 4):
        quad_set = set(quad)

        # Count pairwise connections among the four atoms.
        degree = {a: 0 for a in quad}
        n_edges = 0

        for a, b in combinations(quad, 2):
            if b in neighbor_map[a]:
                n_edges += 1
                degree[a] += 1
                degree[b] += 1

        # First require a pure C4 ring: four edges, degree two at every vertex.
        if not (n_edges == 4 and all(d == 2 for d in degree.values())):
            continue

        # Walk order around the ring: start, one neighbour, the opposite corner,
        # the other neighbour.
        start = quad[0]
        n1, n2 = [x for x in quad if x != start and x in neighbor_map[start]]
        opposite = next(x for x in quad if x not in (start, n1, n2))
        if not ring_closes_locally([start, n1, opposite, n2], positions):
            continue

        # Reject capped/folded C4 motifs:
        # is there any OTHER surface atom directly bonded to all four vertices?
        common_external_neighbors = set.intersection(
            *(set(neighbor_map[a]) for a in quad)
        ) - quad_set

        if common_external_neighbors:
            continue

        sites.add(tuple(sorted(quad)))

    return sorted(sites)

def generate_square_derived_hollow_sites(squares: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Add all triangular subsets of square hollows (including diagonal pairs)."""
    sites = set()
    for sq in squares:
        for tri in combinations(sq, 3):
            sites.add(tuple(sorted(tri)))
    return sorted(sites)


def generate_site_catalog(
    positions: np.ndarray,
    cutoff: float = NI_NEIGHBOR_CUTOFF,
    surface_cn_threshold: int = SURFACE_CN_THRESHOLD,
) -> Tuple[np.ndarray, List[int], Dict[int, set], Dict[str, List[Tuple[int, ...]]]]:
    cn, surface_atoms = detect_surface_atoms(
        positions=positions,
        cutoff=cutoff,
        surface_cn_threshold=surface_cn_threshold,
    )

    neighbor_map = build_neighbor_graph(
        positions=positions,
        atom_indices=surface_atoms,
        cutoff=cutoff,
    )

    square_sites = generate_square_hollow_sites(surface_atoms, neighbor_map, positions) if INCLUDE_HOLLOW4 else []

    standard_hollows = set(generate_hollow_sites(surface_atoms, neighbor_map, positions)) if INCLUDE_HOLLOW else set()
    square_derived   = set(generate_square_derived_hollow_sites(square_sites)) if INCLUDE_SQUARE_DERIVED_HOLLOW else set()
    new_from_squares = square_derived - standard_hollows   # triangles genuinely new (long diagonal)

    site_catalog = {
        "T": generate_top_sites(surface_atoms) if INCLUDE_TOP else [],
        "B": generate_bridge_sites(surface_atoms, neighbor_map) if INCLUDE_BRIDGE else [],
        "H": sorted(standard_hollows | square_derived),
        "F": square_sites,
        "_n_standard_hollows":    len(standard_hollows),
        "_n_square_derived_new":  len(new_from_squares),
    }

    return cn, surface_atoms, neighbor_map, site_catalog


def build_candidate_data(
    positions: np.ndarray,
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    domain_label: str = "NP",
):
    ni_indices = list(range(len(positions)))
    all_anchor_idx = sorted(set(o_anchor_idx) | set(h_anchor_idx))

    local_cutoff = get_local_cutoff(domain_label)
    edge_cutoff = get_edge_cutoff(domain_label)

    # Use EXACTLY the same cell/PBC representation as training
    cell = CURRENT_ATOMS.get_cell()
    pbc = CURRENT_ATOMS.get_pbc()

    if domain_label == "SLAB":
        # Slabs use explicit periodic images, matching the energy training graph.
        # Anchor images are resolved here from the candidate site geometry so
        # that the graph encodes the same O/H arrangement as site_site_distance.
        o_shifts, h_shifts, intra_site_tie, alt_h_shifts = resolve_candidate_anchor_shifts(
            tuple(o_anchor_idx), tuple(h_anchor_idx), positions,
        )
        images, x, local_positions, keep_ni = build_slab_graph_with_shifts(
            energy_module, ni_indices, positions, o_anchor_idx, h_anchor_idx,
            o_shifts, h_shifts, cell, pbc, local_cutoff, edge_cutoff,
        )
        # Bare anchor indices do not uniquely identify a physical candidate if
        # either site has an intra-site image tie. An equally close alternative
        # O/H relative image is only ambiguous if it changes the graph: on a
        # symmetric slab the tied images are mirror copies and give the same
        # input, so the model prediction does not depend on the choice.
        pbc_ambiguous = bool(intra_site_tie)
        if alt_h_shifts and not pbc_ambiguous:
            for alt_shifts in alt_h_shifts:
                _, x_alt, local_positions_alt, _ = build_slab_graph_with_shifts(
                    energy_module, ni_indices, positions, o_anchor_idx, h_anchor_idx,
                    o_shifts, alt_shifts, cell, pbc, local_cutoff, edge_cutoff,
                )
                if not same_slab_graph(
                    x, local_positions, x_alt, local_positions_alt, edge_cutoff,
                ):
                    pbc_ambiguous = True
                    break
        edge_index, edge_attr = energy_module.build_edges_and_edge_attr(
            local_positions, cell, [False, False, False], edge_cutoff,
        )
    else:
        pbc_ambiguous = False
        # NP graphs use the exact feature and edge construction from training.
        keep_ni = energy_module.select_bare_local_environment(
            ni_indices=ni_indices,
            positions=positions,
            anchor_indices=all_anchor_idx,
            cell=cell,
            pbc=pbc,
            cutoff=local_cutoff,
        )

        local_positions = energy_module.unwrap_positions_around_reference(
            positions=positions,
            indices=keep_ni,
            cell=cell,
            pbc=pbc,
            ref_idx=all_anchor_idx[0],
        )

        x = energy_module.build_node_features(
            keep_indices=keep_ni,
            positions=positions,
            o_anchor_idx=o_anchor_idx,
            h_anchor_idx=h_anchor_idx,
            cell=cell,
            pbc=pbc,
            edge_cutoff=edge_cutoff,
        )

        edge_index, edge_attr = energy_module.build_edges_and_edge_attr(
            local_positions=local_positions,
            cell=cell,
            pbc=pbc,
            cutoff=edge_cutoff,
        )

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=torch.tensor(local_positions, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # These are only metadata / candidate diagnostics
    cross_dists = compute_cross_anchor_distances(
        o_anchor_idx=o_anchor_idx,
        h_anchor_idx=h_anchor_idx,
        positions=positions,
    )

    site_distance = site_site_distance(
        tuple(o_anchor_idx),
        tuple(h_anchor_idx),
        positions,
    )

    meta = {
        "o_anchor_idx": list(map(int, o_anchor_idx)),
        "h_anchor_idx": list(map(int, h_anchor_idx)),
        "all_anchor_idx": list(map(int, all_anchor_idx)),
        "keep_ni": list(map(int, keep_ni)),
        "motif": motif_from_anchor_counts(
            len(o_anchor_idx),
            len(h_anchor_idx),
        ),
        "o_candidate_representation": (
            "hybrid_F_four_anchor_candidate"
            if len(o_anchor_idx) == O_FOLDED_N_TOTAL
            else "standard_TBH_candidate"
        ),
        "h_candidate_representation": (
            "hybrid_F_four_anchor_candidate"
            if len(h_anchor_idx) == H_FOLDED_N_TOTAL
            else "standard_TBH_candidate"
        ),
        "shared_anchor_count": len(
            set(o_anchor_idx) & set(h_anchor_idx)
        ),
        "site_distance": float(site_distance),
        "pbc_image_ambiguous": bool(pbc_ambiguous),
        "min_cross_dist": float(np.min(cross_dists)),
        "mean_cross_dist": float(np.mean(cross_dists)),
        "max_cross_dist": float(np.max(cross_dists)),
    }

    return data, meta

def predict_candidate(model, data: Data) -> float:
    data = data.to(DEVICE)
    batch_vec = torch.zeros(data.x.shape[0], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        e_pred = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            batch_vec,
        )
    return float(e_pred.item())

def _support_distance(atoms, i: int, j: int) -> float:
    return float(atoms.get_distance(i, j, mic=bool(np.any(atoms.get_pbc()))))


def _support_mic_vector(atoms, vec: np.ndarray) -> np.ndarray:
    if not np.any(atoms.get_pbc()):
        return np.asarray(vec, dtype=float)
    v_mic, _ = find_mic(np.asarray(vec, dtype=float), cell=atoms.get_cell(), pbc=atoms.get_pbc())
    return np.asarray(v_mic, dtype=float)


def _support_site_position(atoms, site: Tuple[int, ...]) -> np.ndarray:
    positions = atoms.get_positions()
    site = tuple(site)
    ref = site[0]
    ref_pos = positions[ref]
    coords = []
    for idx in site:
        coords.append(ref_pos + _support_mic_vector(atoms, positions[idx] - ref_pos))
    return np.asarray(coords, dtype=float).mean(axis=0)


def _support_site_distance(atoms, site_a: Tuple[int, ...], site_b: Tuple[int, ...]) -> float:
    a = _support_site_position(atoms, site_a)
    b = _support_site_position(atoms, site_b)
    return float(np.linalg.norm(_support_mic_vector(atoms, b - a)))


def _support_find_adsorbate_indices(atoms) -> Tuple[int, int, int]:
    symbols = atoms.get_chemical_symbols()
    o_indices = [i for i, s in enumerate(symbols) if s == "O"]
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]

    if len(o_indices) != 1:
        raise ValueError(f"Expected exactly 1 O, found {len(o_indices)}")
    if len(h_indices) != 2:
        raise ValueError(f"Expected exactly 2 H, found {len(h_indices)}")

    o_idx = o_indices[0]
    dists = sorted([(h, _support_distance(atoms, o_idx, h)) for h in h_indices], key=lambda x: x[1])
    h_oh_idx = dists[0][0]
    h_ads_idx = dists[1][0]
    return o_idx, h_oh_idx, h_ads_idx


def _support_extract_o_anchors_hybrid(
    atoms,
    ni_indices: List[int],
    o_idx: int,
) -> Tuple[List[int], str, Dict[str, object]]:
    """Rebuild O anchors exactly as in the hybrid O/H training scripts."""
    extended_hits = []

    for ni_idx in ni_indices:
        d = _support_distance(atoms, ni_idx, o_idx)
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
        o_site = classify_site_extended(len(selected_hits))
        rule = "standard_short_cutoff"

    return (
        [idx for idx, _ in selected_hits],
        o_site,
        {
            "o_anchor_rule": rule,
            "o_short_contact_count": int(len(short_hits)),
            "o_extended_contact_count": int(len(extended_hits)),
        },
    )

def _support_extract_h_anchors_hybrid(
    atoms,
    ni_indices: List[int],
    h_ads_idx: int,
) -> Tuple[List[int], str, Dict[str, object]]:
    """Rebuild H anchors exactly as in the hybrid training scripts."""
    extended_hits = []

    for ni_idx in ni_indices:
        d = _support_distance(atoms, ni_idx, h_ads_idx)
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
        h_site = classify_site_extended(len(selected_hits))
        rule = "standard_short_cutoff"

    return (
        [idx for idx, _ in selected_hits],
        h_site,
        {
            "h_anchor_rule": rule,
            "h_short_contact_count": int(len(short_hits)),
            "h_extended_contact_count": int(len(extended_hits)),
        },
    )


def compute_training_support(extxyz_files):
    """Build motif-specific geometric reference ranges from train + val frames."""
    support_raw = defaultdict(list)

    for filename in extxyz_files:
        if filename is None:
            continue
        frames = read(filename, index=":")

        for atoms in frames:
            symbols = atoms.get_chemical_symbols()
            ni_indices = [i for i, s in enumerate(symbols) if s == "Ni"]

            o_idx, _, h_ads_idx = _support_find_adsorbate_indices(atoms)

            o_anchor_idx, o_site, o_anchor_meta = _support_extract_o_anchors_hybrid(
                atoms,
                ni_indices,
                o_idx,
            )
            h_anchor_idx, h_site, h_anchor_meta = _support_extract_h_anchors_hybrid(
                atoms,
                ni_indices,
                h_ads_idx,
            )

            if len(o_anchor_idx) == 0 or len(h_anchor_idx) == 0:
                continue

            motif = o_site + h_site

            cross_dists = []
            for i in o_anchor_idx:
                for j in h_anchor_idx:
                    cross_dists.append(_support_distance(atoms, i, j))
            cross_dists = np.asarray(cross_dists, dtype=float)

            site_distance = _support_site_distance(atoms, tuple(o_anchor_idx), tuple(h_anchor_idx))

            support_raw[motif].append({
                "site_distance": site_distance,
                "min_cross_dist": float(np.min(cross_dists)),
                "mean_cross_dist": float(np.mean(cross_dists)),
                "max_cross_dist": float(np.max(cross_dists)),
                "n_o": len(o_anchor_idx),
                "n_h": len(h_anchor_idx),
                "o_anchor_rule": o_anchor_meta["o_anchor_rule"],
                "o_short_contact_count": o_anchor_meta["o_short_contact_count"],
                "o_extended_contact_count": o_anchor_meta["o_extended_contact_count"],
                "h_anchor_rule": h_anchor_meta["h_anchor_rule"],
                "h_short_contact_count": h_anchor_meta["h_short_contact_count"],
                "h_extended_contact_count": h_anchor_meta["h_extended_contact_count"],
            })

    support = {}

    for motif, rows in support_raw.items():
        support[motif] = {}

        for key in ["site_distance", "min_cross_dist", "mean_cross_dist", "max_cross_dist"]:
            vals = np.array([r[key] for r in rows], dtype=float)
            lo, hi = np.percentile(vals, SUPPORT_PERCENTILES)
            support[motif][key] = (float(lo), float(hi))

        support[motif]["count"] = len(rows)
        support[motif]["o_anchor_rule_counts"] = dict(
            (rule, sum(r["o_anchor_rule"] == rule for r in rows))
            for rule in sorted(set(r["o_anchor_rule"] for r in rows))
        )
        support[motif]["h_anchor_rule_counts"] = dict(
            (rule, sum(r["h_anchor_rule"] == rule for r in rows))
            for rule in sorted(set(r["h_anchor_rule"] for r in rows))
        )

    return support

# ============================================================
# MIC-aware site geometry and classifier-graph helpers
# ============================================================

def _pbc_active() -> bool:
    return bool(USE_PBC and CURRENT_ATOMS is not None and np.any(CURRENT_ATOMS.get_pbc()))


def _cell_pbc():
    if CURRENT_ATOMS is None:
        return None, None
    return CURRENT_ATOMS.get_cell(), CURRENT_ATOMS.get_pbc()


def mic_vector(vec: np.ndarray) -> np.ndarray:
    """Return minimum-image version of a Cartesian vector."""
    if not _pbc_active():
        return np.asarray(vec, dtype=float)
    cell, pbc = _cell_pbc()
    v_mic, _ = find_mic(np.asarray(vec, dtype=float), cell=cell, pbc=pbc)
    return np.asarray(v_mic, dtype=float)


def mic_distance_between_positions(pos_i: np.ndarray, pos_j: np.ndarray) -> float:
    return float(np.linalg.norm(mic_vector(np.asarray(pos_j) - np.asarray(pos_i))))


def unwrap_indices_around_reference(indices: List[int], positions: np.ndarray, ref_idx: int | None = None) -> np.ndarray:
    """Return local Cartesian coordinates unwrapped around ref_idx.

    This is important for sites or local graphs crossing a periodic boundary.
    The first/ref atom stays at its original Cartesian position and all other
    atoms are placed using MIC displacement vectors from that reference.
    """
    indices = list(indices)
    if not indices:
        return np.zeros((0, 3), dtype=float)
    if ref_idx is None:
        ref_idx = indices[0]
    ref_pos = np.asarray(positions[ref_idx], dtype=float)
    coords = []
    for idx in indices:
        disp = mic_vector(np.asarray(positions[idx], dtype=float) - ref_pos)
        coords.append(ref_pos + disp)
    return np.asarray(coords, dtype=float)


def compute_site_position(site: Tuple[int, ...], positions: np.ndarray) -> np.ndarray:
    """Centroid of an anchor site, PBC-unwrapped if needed."""
    site = tuple(site)
    if len(site) == 0:
        return np.zeros(3, dtype=float)
    coords = unwrap_indices_around_reference(list(site), positions, ref_idx=site[0])
    return coords.mean(axis=0)


# Two periodic images whose lengths differ only at numerical precision are
# treated as a true tie: the index-based site definition cannot say which one
# is meant. Across the supplied slab systems, genuine ties differ by at most
# 5e-9 A, whereas the smallest non-tied gap is 1.59 A.
PBC_IMAGE_TIE_TOL = 1.0e-6  # Angstrom


def _image_margin(vec: np.ndarray) -> float:
    """Length gap between the shortest and second-shortest periodic image of vec.

    A small gap means the minimum-image choice is effectively arbitrary (e.g. a
    nearest-neighbour pair separated by exactly half a lattice vector, as in a
    Ni(111)-2x2 cell).
    """
    if not _pbc_active():
        return float("inf")
    cell, pbc = _cell_pbc()
    cell_arr = np.asarray(cell)
    pbc_arr = np.asarray(pbc, dtype=bool)
    inv_cell = np.linalg.inv(cell_arr)
    frac = -np.asarray(vec, dtype=float) @ inv_cell
    center = np.where(pbc_arr, np.round(frac), 0.0).astype(int)
    offsets = [range(-1, 2) if periodic else range(1) for periodic in pbc_arr]
    lengths = sorted(
        float(
            np.linalg.norm(
                np.asarray(vec, dtype=float)
                + (center + np.array([a, b, c])) @ cell_arr
            )
        )
        for a in offsets[0] for b in offsets[1] for c in offsets[2]
    )
    return lengths[1] - lengths[0] if len(lengths) > 1 else float("inf")


def resolve_candidate_anchor_shifts(
    o_site: Tuple[int, ...],
    h_site: Tuple[int, ...],
    positions: np.ndarray,
):
    """Integer cell shifts placing each candidate anchor in one consistent image.

    Each site is unwrapped around its first anchor (as in compute_site_position),
    and the H site is then moved rigidly so that its centroid sits at
    ``c_O + MIC(c_H - c_O)``. This is exactly the O/H arrangement used by
    site_site_distance() for candidate filtering, so the graph encodes the same
    pair that passed the distance filters.

    Returns (o_shifts, h_shifts, intra_site_tie, alt_h_shifts):
      intra_site_tie -- an anchor pair inside one site has two nearly equally
                        short images, so the site geometry itself is undefined;
      alt_h_shifts   -- H shift maps for other H-site images whose centroid is
                        (within PBC_IMAGE_TIE_TOL) as close to the O site.
    """
    # Anchor tuples describe unordered sites. Canonicalizing them prevents the
    # arbitrary caller order from selecting a different reference atom/image.
    o_site = tuple(sorted(o_site))
    h_site = tuple(sorted(h_site))
    zero = (0, 0, 0)
    if not _pbc_active():
        return {b: zero for b in o_site}, {b: zero for b in h_site}, False, []

    cell, pbc = _cell_pbc()
    cell_arr = np.asarray(cell)
    inv_cell = np.linalg.inv(cell_arr)
    pbc_arr = np.asarray(pbc, dtype=bool)
    intra_site_tie = False

    def _to_shift(cart_shift):
        frac = np.asarray(cart_shift) @ inv_cell
        return tuple(int(v) for v in np.where(pbc_arr, np.round(frac), 0))

    def _unwrap(site):
        nonlocal intra_site_tie
        ref_pos = np.asarray(positions[site[0]], dtype=float)
        shifts, coords = {}, []
        for b in site:
            raw = np.asarray(positions[b], dtype=float) - ref_pos
            if b != site[0] and _image_margin(raw) < PBC_IMAGE_TIE_TOL:
                intra_site_tie = True
            u = ref_pos + mic_vector(raw)
            shifts[b] = _to_shift(u - np.asarray(positions[b], dtype=float))
            coords.append(u)
        return shifts, np.mean(coords, axis=0)

    o_shifts, c_o = _unwrap(o_site)
    h_shifts, c_h = _unwrap(h_site)

    v = c_h - c_o
    v_mic = mic_vector(v)
    rigid = np.array(_to_shift((c_o + v_mic) - c_h))
    best = float(np.linalg.norm(v_mic))
    ranges = [range(-1, 2) if p else range(1) for p in pbc_arr]
    alt_rigids = []
    for a in ranges[0]:
        for b in ranges[1]:
            for c in ranges[2]:
                if a == b == c == 0:
                    continue
                cand = rigid + np.array([a, b, c])
                d = float(np.linalg.norm(v + cand @ cell_arr))
                if d - best < PBC_IMAGE_TIE_TOL:
                    alt_rigids.append(tuple(int(x) for x in cand))

    def _apply(r):
        return {b: tuple(x + y for x, y in zip(s, r)) for b, s in h_shifts.items()}

    return o_shifts, _apply(tuple(int(x) for x in rigid)), intra_site_tie, [_apply(r) for r in alt_rigids]


def _slab_graph_api(module) -> str:
    """Identify how ``module.build_slab_local_graph`` accepts anchor images.

    "keyword_shifts" -- positional (..., anchor_positions, cell, pbc,
                        local_cutoff, edge_cutoff) plus optional keyword
                        o_shifts/h_shifts; used when supplied.
    "positional_shifts" -- (..., o_shifts, h_shifts, cell, pbc, ...), shifts
                        always used.
    "legacy"         -- no shift arguments; images come from the module-level
                        resolve_anchor_shifts(), which can be overridden.
    Anything else is rejected with an explicit error rather than guessed.
    """
    fn = getattr(module, "build_slab_local_graph", None)
    if fn is None:
        return "none"
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return "unknown"
    names = list(params)
    if "o_shifts" in params and "h_shifts" in params:
        if names.index("o_shifts") == 4 and names.index("h_shifts") == 5:
            return "positional_shifts"
        if params["o_shifts"].default is None and params["h_shifts"].default is None:
            return "keyword_shifts"
        return "unknown"
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = ""
    if len(names) >= 9 and "resolve_anchor_shifts(" in src and hasattr(module, "resolve_anchor_shifts"):
        return "legacy"
    return "unknown"


def build_slab_graph_with_shifts(
    module, ni_indices, positions, o_anchor_idx, h_anchor_idx,
    o_shifts, h_shifts, cell, pbc, local_cutoff, edge_cutoff,
):
    """Call ``module.build_slab_local_graph`` with explicitly resolved shifts."""
    api = _slab_graph_api(module)
    if api == "keyword_shifts":
        return module.build_slab_local_graph(
            ni_indices, positions, o_anchor_idx, h_anchor_idx,
            positions, cell, pbc, local_cutoff, edge_cutoff,
            o_shifts=o_shifts, h_shifts=h_shifts,
        )
    if api == "positional_shifts":
        return module.build_slab_local_graph(
            ni_indices, positions, o_anchor_idx, h_anchor_idx,
            o_shifts, h_shifts, cell, pbc, local_cutoff, edge_cutoff,
        )
    if api == "legacy":
        original = module.resolve_anchor_shifts
        module.resolve_anchor_shifts = lambda *args, **kwargs: (o_shifts, h_shifts)
        try:
            return module.build_slab_local_graph(
                ni_indices, positions, o_anchor_idx, h_anchor_idx,
                positions, cell, pbc, local_cutoff, edge_cutoff,
            )
        finally:
            module.resolve_anchor_shifts = original
    try:
        where = inspect.getsourcefile(module)
        sig = str(inspect.signature(module.build_slab_local_graph))
    except (TypeError, ValueError, AttributeError):
        where, sig = getattr(module, "__file__", "?"), "?"
    raise RuntimeError(
        f"Unsupported build_slab_local_graph in {module.__name__} ({where}).\n"
        f"  signature: {sig}"
    )


def same_slab_graph(x_a, pos_a, x_b, pos_b, edge_cutoff: float, atol: float = 1.0e-3) -> bool:
    """True if two slab graphs have the same node features and edge lengths.

    Node-feature rows and Ni-Ni edge lengths are compared as sorted multisets,
    so graphs that differ only by a rigid motion or a mirror operation (as for
    tied periodic images on a symmetric slab) compare equal.
    """
    def _rows(x):
        x = np.round(np.asarray(x, dtype=float), 4)
        return x[np.lexsort(x.T[::-1])]

    def _edges(pos):
        pos = np.asarray(pos, dtype=float)
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        iu = np.triu_indices(len(pos), k=1)
        d = d[iu]
        return np.sort(d[d <= edge_cutoff])

    rows_a, rows_b = _rows(x_a), _rows(x_b)
    if rows_a.shape != rows_b.shape or not np.allclose(rows_a, rows_b, atol=atol):
        return False
    edges_a, edges_b = _edges(pos_a), _edges(pos_b)
    return edges_a.shape == edges_b.shape and np.allclose(edges_a, edges_b, atol=atol)


def site_site_distance(o_site: Tuple[int, ...], h_site: Tuple[int, ...], positions: np.ndarray) -> float:
    """Distance between two site centroids using MIC between the centroids."""
    o_pos = compute_site_position(o_site, positions)
    h_pos = compute_site_position(h_site, positions)
    return float(np.linalg.norm(mic_vector(h_pos - o_pos)))


def sites_too_close(
    o_site: Tuple[int, ...],
    h_site: Tuple[int, ...],
    positions: np.ndarray,
    min_dist: float = MIN_SITE_DIST,
    max_dist: float = MAX_SITE_DIST,
) -> bool:
    d = site_site_distance(o_site, h_site, positions)
    return (d < min_dist) or (d > max_dist)


def compute_coordination_numbers_pbc(
    positions: np.ndarray,
    atom_indices: List[int] | None = None,
    cutoff: float = NI_NEIGHBOR_CUTOFF,
) -> np.ndarray:
    """Coordination numbers using MIC distances when PBC is active."""
    n = len(positions)
    if atom_indices is None:
        atom_indices = list(range(n))
    cn = np.zeros(n, dtype=int)
    for i, j in combinations(atom_indices, 2):
        d = mic_distance_between_positions(positions[i], positions[j])
        if d <= cutoff:
            cn[i] += 1
            cn[j] += 1
    # Atoms outside atom_indices remain zero by construction.
    return cn


def compute_surface_atoms(
    positions: np.ndarray,
    cutoff: float = NI_NEIGHBOR_CUTOFF,
    surface_cn_threshold: int = SURFACE_CN_THRESHOLD,
) -> Tuple[np.ndarray, List[int]]:
    """CN-based surface atoms, optionally restricted to the top slab side."""
    cn = compute_coordination_numbers_pbc(positions, cutoff=cutoff)
    surface_atoms = [i for i in range(len(positions)) if int(cn[i]) < surface_cn_threshold]

    if _pbc_active() and TOP_SURFACE_ONLY and len(surface_atoms):
        z = positions[:, TOP_AXIS]
        z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))
        surface_atoms = [i for i in surface_atoms if positions[i, TOP_AXIS] >= z_mid]

    return cn, surface_atoms


def mic_lengths(vectors: np.ndarray) -> np.ndarray:
    """MIC-aware norms of an (N, 3) array of Cartesian vectors.

    Falls back to plain Euclidean norms when PBC is not active, so it works for
    isolated NPs and for slabs alike.
    """
    vectors = np.asarray(vectors, dtype=float)
    if not _pbc_active():
        return np.linalg.norm(vectors, axis=1)
    cell, pbc = _cell_pbc()
    _, lengths = find_mic(vectors, cell=cell, pbc=pbc)
    return np.asarray(lengths, dtype=float)


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """Roughly uniform points on the unit sphere (golden-spiral construction)."""
    i = np.arange(n_points)
    z = 1.0 - 2.0 * (i + 0.5) / n_points
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i  # golden angle
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def compute_surface_atoms_exposure(
    positions: np.ndarray,
    cutoff: float = NI_NEIGHBOR_CUTOFF,
    r_atom: float = SURFACE_R_ATOM,
    r_probe: float = SURFACE_R_PROBE,
    n_points: int = SURFACE_N_POINTS,
    min_exposed_fraction: float = SURFACE_MIN_EXPOSED_FRACTION,
) -> Tuple[np.ndarray, List[int]]:
    """Select Ni atoms accessible to a spherical probe.

    A surface atom has at least one unburied point on its probe-expanded
    sphere. Return CN as well for reporting, even in exposure mode.
    """
    n = len(positions)
    cn = compute_coordination_numbers_pbc(positions, cutoff=cutoff)  # for reporting only
    sphere = _fibonacci_sphere(n_points)
    shell = r_atom + r_probe
    surface_atoms: List[int] = []

    for i in range(n):
        pts = positions[i] + shell * sphere          # test points on atom i's shell
        buried = np.zeros(n_points, dtype=bool)
        for j in range(n):
            if j == i:
                continue
            # Atoms farther than two shells cannot bury any test point.
            if mic_distance_between_positions(positions[i], positions[j]) > 2.0 * shell:
                continue
            buried |= mic_lengths(pts - positions[j]) < shell
            if buried.all():
                break
        if np.mean(~buried) > min_exposed_fraction:
            surface_atoms.append(i)

    if _pbc_active() and TOP_SURFACE_ONLY and len(surface_atoms):
        z = positions[:, TOP_AXIS]
        z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))
        surface_atoms = [i for i in surface_atoms if positions[i, TOP_AXIS] >= z_mid]

    return cn, surface_atoms


def detect_surface_atoms(
    positions: np.ndarray,
    cutoff: float = NI_NEIGHBOR_CUTOFF,
    surface_cn_threshold: int = SURFACE_CN_THRESHOLD,
) -> Tuple[np.ndarray, List[int]]:
    """Dispatch surface detection according to SURFACE_METHOD."""
    if SURFACE_METHOD == "exposure":
        return compute_surface_atoms_exposure(
            positions=positions,
            cutoff=cutoff,
            r_atom=SURFACE_R_ATOM,
            r_probe=SURFACE_R_PROBE,
            n_points=SURFACE_N_POINTS,
            min_exposed_fraction=SURFACE_MIN_EXPOSED_FRACTION,
        )
    if SURFACE_METHOD == "cn":
        return compute_surface_atoms(
            positions=positions,
            cutoff=cutoff,
            surface_cn_threshold=surface_cn_threshold,
        )
    raise ValueError(
        f"Unknown SURFACE_METHOD={SURFACE_METHOD!r}; expected 'exposure' or 'cn'."
    )


def build_neighbor_graph(
    positions: np.ndarray,
    atom_indices: List[int],
    cutoff: float = NI_NEIGHBOR_CUTOFF,
) -> Dict[int, set]:
    nbrs = {i: set() for i in atom_indices}
    for i, j in combinations(atom_indices, 2):
        d = mic_distance_between_positions(positions[i], positions[j])
        if d <= cutoff:
            nbrs[i].add(j)
            nbrs[j].add(i)
    return nbrs


def select_bare_local_environment_pbc(
    ni_indices: List[int],
    positions: np.ndarray,
    anchor_indices: List[int],
    cutoff: float,
) -> List[int]:
    keep = []
    for i in ni_indices:
        for a in anchor_indices:
            if mic_distance_between_positions(positions[i], positions[a]) <= cutoff:
                keep.append(i)
                break
    return sorted(set(keep))


def build_edges_and_edge_attr_pbc(
    local_positions: np.ndarray,
    cutoff: float = EDGE_CUTOFF,
):
    """Create a local Euclidean Ni graph with Gaussian RBF edge features."""
    edge_src = []
    edge_dst = []
    dists = []
    n = len(local_positions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(local_positions[j] - local_positions[i]))
            if d <= cutoff:
                edge_src.append(i)
                edge_dst.append(j)
                dists.append(d)

    if len(edge_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, RBF_N_CENTERS), dtype=torch.float)
        return edge_index, edge_attr

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    centers = np.linspace(RBF_R_MIN, RBF_R_MAX, RBF_N_CENTERS, dtype=np.float32)
    d = np.asarray(dists, dtype=np.float32)[:, None]
    edge_attr_np = np.exp(-RBF_GAMMA * (d - centers[None, :]) ** 2)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
    return edge_index, edge_attr


def build_node_features_pbc(
    keep_indices: List[int],
    positions: np.ndarray,
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    edge_cutoff: float = EDGE_CUTOFF,
) -> np.ndarray:
    """Build the classifier's eight node features on an unwrapped NP patch."""
    keep_indices = list(keep_indices)
    if not keep_indices:
        return np.zeros((0, 8), dtype=np.float32)

    ref = sorted(set(o_anchor_idx) | set(h_anchor_idx))[0]
    local_positions = unwrap_indices_around_reference(keep_indices, positions, ref_idx=ref)
    local_pos_by_global = {idx: local_positions[k] for k, idx in enumerate(keep_indices)}

    def centroid(anchor_idx: List[int]) -> np.ndarray:
        coords = []
        for a in anchor_idx:
            if a in local_pos_by_global:
                coords.append(local_pos_by_global[a])
            else:
                coords.append(positions[ref] + mic_vector(positions[a] - positions[ref]))
        return np.asarray(coords, dtype=float).mean(axis=0)

    o_cent = centroid(o_anchor_idx)
    h_cent = centroid(h_anchor_idx)
    all_cent = centroid(sorted(set(o_anchor_idx) | set(h_anchor_idx)))

    # CN inside the selected local subgraph, matching the local-CN training setup.
    local_cn = np.zeros(len(keep_indices), dtype=float)
    for i, j in combinations(range(len(keep_indices)), 2):
        d = float(np.linalg.norm(local_positions[j] - local_positions[i]))
        if d <= edge_cutoff:
            local_cn[i] += 1
            local_cn[j] += 1

    feats = []
    o_set = set(o_anchor_idx)
    h_set = set(h_anchor_idx)
    for k, idx in enumerate(keep_indices):
        pos = local_positions[k]
        is_o = 1.0 if idx in o_set else 0.0
        is_h = 1.0 if idx in h_set else 0.0
        is_any = 1.0 if (is_o or is_h) else 0.0
        feats.append([
            1.0,
            is_o,
            is_h,
            is_any,
            float(np.linalg.norm(pos - o_cent)),
            float(np.linalg.norm(pos - h_cent)),
            float(np.linalg.norm(pos - all_cent)),
            float(local_cn[k]),
        ])
    return np.asarray(feats, dtype=np.float32)


def compute_cross_anchor_distances(
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    positions: np.ndarray,
) -> np.ndarray:
    dists = []
    for i in o_anchor_idx:
        for j in h_anchor_idx:
            dists.append(mic_distance_between_positions(positions[i], positions[j]))
    return np.array(dists, dtype=np.float32)


def build_retention_candidate_data(
    positions: np.ndarray,
    o_anchor_idx: List[int],
    h_anchor_idx: List[int],
    cls_module,
    domain_label: str,
):
    """Build the retention-classifier graph using its domain-specific cutoffs."""
    ni_indices = list(range(len(positions)))
    all_anchor_idx = sorted(set(o_anchor_idx) | set(h_anchor_idx))

    local_cutoff = float(cls_module.get_local_cutoff(domain_label))
    edge_cutoff = float(cls_module.get_edge_cutoff(domain_label))

    if domain_label == "SLAB":
        # Retention slabs also use their training module's explicit-image graph.
        cell = CURRENT_ATOMS.get_cell()
        pbc = CURRENT_ATOMS.get_pbc()

        o_shifts, h_shifts, _, _ = resolve_candidate_anchor_shifts(
            tuple(o_anchor_idx), tuple(h_anchor_idx), positions,
        )
        images, x, local_positions, keep_ni = build_slab_graph_with_shifts(
            cls_module, ni_indices, positions, o_anchor_idx, h_anchor_idx,
            o_shifts, h_shifts, cell, pbc, local_cutoff, edge_cutoff,
        )
        edge_index, edge_attr = cls_module.build_edges_and_edge_attr(
            local_positions, cell, [False, False, False], edge_cutoff,
        )
    else:
        keep_ni = select_bare_local_environment_pbc(
            ni_indices=ni_indices,
            positions=positions,
            anchor_indices=all_anchor_idx,
            cutoff=local_cutoff,
        )
        ref = all_anchor_idx[0]
        local_positions = unwrap_indices_around_reference(keep_ni, positions, ref_idx=ref)
        x = build_node_features_pbc(
            keep_indices=keep_ni,
            positions=positions,
            o_anchor_idx=o_anchor_idx,
            h_anchor_idx=h_anchor_idx,
            edge_cutoff=edge_cutoff,
        )
        edge_index, edge_attr = build_edges_and_edge_attr_pbc(
            local_positions=local_positions,
            cutoff=edge_cutoff,
        )

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=torch.tensor(local_positions, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

def check_candidate_support(meta, training_support):
    """Return geometric support flags; never filter or re-rank candidates."""
    motif = meta["motif"]

    if motif not in training_support:
        return {
            "seen_motif": False,
            "support_count": 0,
            "support_o_anchor_rule_counts": {},
            "support_h_anchor_rule_counts": {},
            "ood_score": 999,
            "ood_reason": "motif_not_seen",
        }

    bounds = training_support[motif]
    ood_score = 0
    reasons = []

    for key in ["min_cross_dist", "mean_cross_dist", "max_cross_dist"]:
        if key not in meta or key not in bounds:
            continue

        val = float(meta[key])
        lo, hi = bounds[key]

        if val < lo or val > hi:
            ood_score += 1
            reasons.append(f"{key}_out")

    if len(reasons) == 0:
        reason = "in_training_support"
    else:
        reason = ";".join(reasons)

    return {
        "seen_motif": True,
        "support_count": int(bounds.get("count", 0)),
        "support_o_anchor_rule_counts": bounds.get("o_anchor_rule_counts", {}),
        "support_h_anchor_rule_counts": bounds.get("h_anchor_rule_counts", {}),
        "ood_score": int(ood_score),
        "ood_reason": reason,
    }

# ============================================================
# MAIN
# ============================================================

def report_graph_representation(module, label: str) -> None:
    """Report how slab periodic images are resolved for this module."""
    api = _slab_graph_api(module)
    if api == "none":
        print(f"{label} slab representation: unique-atom + MIC (no explicit-image graph)")
    elif api in ("keyword_shifts", "positional_shifts"):
        print(f"{label} slab representation: explicit periodic images; anchor shifts "
              "supplied by prediction.py (site-geometry resolution)")
    elif api == "legacy":
        print(f"{label} slab representation: original module; its resolve_anchor_shifts is "
              "overridden during graph construction with prediction.py's site-geometry shifts")
    else:
        print(f"WARNING: {label} slab representation: unsupported build_slab_local_graph "
              "in the imported module; slab inputs will stop with an explanatory error.")


def main():
    validate_user_settings()
    report_resolved_paths()
    print()

    print(f"Energy training module: {ENERGY_TRAIN_MODULE}")
    print(
        "Energy PBC representation: NP -> unique-atom + MIC selection with "
        "one-reference unwrapping; SLAB -> explicit periodic-image graph from "
        "the training module (build_slab_local_graph), then Euclidean "
        "local CN/node distances/edges"
    )
    report_graph_representation(energy_module, "Energy")
    model, ckpt = load_model(MODEL_PATH)
    print("Energy objective: energy only")
    print("Predicted-energy reference: H2O(g)")

    print("\n=== HYBRID O/H-ANCHOR DEFINITIONS ===")
    print(f"O standard T/B/H cutoff: Ni-O < {NI_O_CUTOFF:.3f} Å")
    print(
        "O folded F rule: "
        f"at least {O_FOLDED_N_SHORT} contacts below {NI_O_CUTOFF:.3f} Å "
        f"and exactly {O_FOLDED_N_TOTAL} contacts below "
        f"{NI_O_FOLDED_CUTOFF:.3f} Å"
    )
    print(f"H standard T/B/H cutoff: Ni-H < {NI_H_CUTOFF:.3f} Å")
    print(
        "Folded F rule: "
        f"at least {H_FOLDED_N_SHORT} contacts below {NI_H_CUTOFF:.3f} Å "
        f"and exactly {H_FOLDED_N_TOTAL} contacts below "
        f"{NI_H_FOLDED_CUTOFF:.3f} Å"
    )
    print(
        "Bare inference note: F candidates are generated geometrically as "
        "four-anchor square/folded sites; the at-least-two-short and "
        "exactly-four-total condition is used when reconstructing adsorbed "
        "train/validation support."
    )

    retention_model = None
    retention_ckpt = None
    retention_module = None
    if USE_RETENTION_MODEL:
        retention_model, retention_ckpt, retention_module = load_retention_model(
            RETENTION_MODEL_PATH,
            RETENTION_TRAIN_MODULE,
        )
        print("\n=== MOTIF-RETENTION CLASSIFIER ===")
        print(f"Loaded: {RETENTION_MODEL_PATH}")
        print(f"Training module: {RETENTION_TRAIN_MODULE}")
        report_graph_representation(retention_module, "Retention")
        print("Task: binary exact-site retention (NP/SLAB heads)")
        if "split_mode" in retention_ckpt:
            print(f"split_mode: {retention_ckpt['split_mode']}")
        if "early_stopping_metric" in retention_ckpt:
            print(f"early_stopping_metric: {retention_ckpt['early_stopping_metric']}")
        print(
            "Thresholds: "
            f"screening={RETENTION_SCREENING_THRESHOLD}, "
            f"conservative={RETENTION_CONSERVATIVE_THRESHOLD}, "
            f"highconf={RETENTION_HIGHCONF_THRESHOLD}"
        )

    training_support = compute_training_support([
        TRAIN_SUPPORT_EXTXYZ,
        VAL_SUPPORT_EXTXYZ,
    ])

    print("\n=== TRAINING SUPPORT BY MOTIF ===")

    for motif, info in sorted(training_support.items()):
        print(
            f"{motif}: n = {info['count']} | "
            f"O rules = {info.get('o_anchor_rule_counts', {})} | "
            f"H rules = {info.get('h_anchor_rule_counts', {})}"
        )

    atoms = read(BARE_CLUSTER_FILE)
    global CURRENT_ATOMS
    CURRENT_ATOMS = atoms
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    if any(sym != "Ni" for sym in symbols):
        raise ValueError("This script expects a bare Ni cluster only.")

    # Detect whether this is a slab (any PBC direction on) or an isolated NP.
    # The corresponding LOCAL_CUTOFF / EDGE_CUTOFF are pulled from the training
    # module's per-domain dictionaries when available.
    domain_label = domain_label_for(atoms)
    print(f"Loaded bare cluster: {BARE_CLUSTER_FILE}")
    print(f"Number of Ni atoms: {len(positions)}")
    print(
        "PBC flags: "
        + " ".join("T" if bool(value) else "F" for value in atoms.get_pbc())
    )
    print(f"Domain detected from PBC: {domain_label}")
    print(f"  LOCAL_CUTOFF used: {get_local_cutoff(domain_label):.3f} Å")
    print(f"  EDGE_CUTOFF  used: {get_edge_cutoff(domain_label):.3f} Å")

    cn, surface_atoms, neighbor_map, site_catalog = generate_site_catalog(
        positions=positions,
        cutoff=NI_NEIGHBOR_CUTOFF,
        surface_cn_threshold=SURFACE_CN_THRESHOLD,
    )

    print("\n=== SURFACE DETECTION ===")
    if SURFACE_METHOD == "exposure":
        print(
            f"Surface criterion: probe-sphere accessibility "
            f"(r_atom={SURFACE_R_ATOM:.2f} Å, r_probe={SURFACE_R_PROBE:.2f} Å, "
            f"{SURFACE_N_POINTS} pts, min_exposed={SURFACE_MIN_EXPOSED_FRACTION:.3f})"
        )
    else:
        print(f"Surface criterion: CN < {SURFACE_CN_THRESHOLD}")
    print(f"Number of surface atoms: {len(surface_atoms)}")
    if PRINT_SURFACE_DETAILS:
        print(f"Surface indices (zero-based): {surface_atoms}")
        for i in surface_atoms:
            print(f"  atom {i} | CN = {int(cn[i])}")

    print("\n=== SITE COUNTS ===")
    print(f"Top sites:                      {len(site_catalog['T'])}")
    print(f"Bridge sites:                   {len(site_catalog['B'])}")
    print(f"Hollow sites (3-fold total):    {len(site_catalog['H'])}")
    print(f"  of which standard (all edges within cutoff): {site_catalog['_n_standard_hollows']}")
    print(f"  of which new from square diagonals:          {site_catalog['_n_square_derived_new']}")
    print(f"Square sites (4-fold):          {len(site_catalog['F'])}")
    print("  Fourfold sites exclude capped/surface-folded squares.")

    results = []
    motif_counts = defaultdict(int)

    candidate_id = 0
    n_skipped_far = 0
    n_skipped_pbc_ambiguous = 0
    site_types = ["T", "B", "H", "F"]

    for o_type in site_types:
        for h_type in site_types:
            for o_site in site_catalog[o_type]:
                for h_site in site_catalog[h_type]:
                    if not candidate_is_allowed(o_site, h_site, positions):
                        continue

                    # Skip far pairs before the expensive model evaluations.
                    if SCORE_NEAR_ONLY:
                        if site_site_distance(
                            tuple(o_site), tuple(h_site), positions
                        ) > PAIR_DISTANCE_SPLIT:
                            n_skipped_far += 1
                            continue

                    data, meta = build_candidate_data(
                        positions=positions,
                        o_anchor_idx=list(o_site),
                        h_anchor_idx=list(h_site),
                        domain_label=domain_label,
                    )

                    if (
                        domain_label == "SLAB"
                        and SKIP_PBC_AMBIGUOUS
                        and meta.get("pbc_image_ambiguous", False)
                    ):
                        n_skipped_pbc_ambiguous += 1
                        continue

                    support_info = check_candidate_support(meta, training_support)

                    e_pred = predict_candidate(model, data)

                    if USE_RETENTION_MODEL:
                        retention_data = build_retention_candidate_data(
                            positions=positions,
                            o_anchor_idx=list(o_site),
                            h_anchor_idx=list(h_site),
                            cls_module=retention_module,
                            domain_label=domain_label,
                        )
                        retention_score = predict_retention_probability(
                            retention_model,
                            retention_data,
                            domain_label=domain_label,
                            cls_module=retention_module,
                        )
                    else:
                        retention_score = float("nan")
                    # Use the distance between O- and H-site centroids.
                    distance_class = (
                        "near" if meta["site_distance"] <= PAIR_DISTANCE_SPLIT else "far"
                    )

                    row = {
                        "candidate_id": candidate_id,
                        "motif": meta["motif"],
                        "o_type": o_type,
                        "h_type": h_type,
                        "o_anchor_idx": meta["o_anchor_idx"],
                        "h_anchor_idx": meta["h_anchor_idx"],
                        "o_candidate_representation": meta["o_candidate_representation"],
                        "h_candidate_representation": meta["h_candidate_representation"],
                        "shared_anchor_count": meta["shared_anchor_count"],
                        "pred_Eads_eV": e_pred,
                        # Binary probability that the exact O/H binding sites are retained.
                        "retention_probability": retention_score,
                        "retention_flag": retention_flag_from_probability(retention_score),
                        "distance_class": distance_class,
                        "min_cross_dist": meta["min_cross_dist"],
                        "mean_cross_dist": meta["mean_cross_dist"],
                        "max_cross_dist": meta["max_cross_dist"],
                        "site_distance": meta.get("site_distance", float("nan")),
                        "pbc_image_ambiguous": meta.get("pbc_image_ambiguous", False),
                        "seen_motif": support_info["seen_motif"],
                        "support_count": support_info["support_count"],
                        "support_o_anchor_rule_counts": support_info["support_o_anchor_rule_counts"],
                        "support_h_anchor_rule_counts": support_info["support_h_anchor_rule_counts"],
                        "ood_score": support_info["ood_score"],
                        "ood_reason": support_info["ood_reason"],
                    }

                    results.append(row)
                    motif_counts[row["motif"]] += 1
                    candidate_id += 1

                    if candidate_id % 1000 == 0:
                        print(f"Scored {candidate_id} candidates")

                    if MAX_CANDIDATES is not None and candidate_id >= MAX_CANDIDATES:
                        break
                if MAX_CANDIDATES is not None and candidate_id >= MAX_CANDIDATES:
                    break
            if MAX_CANDIDATES is not None and candidate_id >= MAX_CANDIDATES:
                break
        if MAX_CANDIDATES is not None and candidate_id >= MAX_CANDIDATES:
            break

    results = sorted(results, key=lambda x: x["pred_Eads_eV"])

    print("\n=== MOTIF COUNTS IN SCORED CANDIDATES ===")
    for motif in sorted(motif_counts.keys()):
        print(f"  {motif}: {motif_counts[motif]}")

    print(f"\nTotal candidates scored: {len(results)}")
    if domain_label == "SLAB" and SKIP_PBC_AMBIGUOUS:
        print(
            "PBC-ambiguous candidates skipped unscored: "
            f"{n_skipped_pbc_ambiguous}"
        )
    amb_ids = sorted(r["candidate_id"] for r in results if r["pbc_image_ambiguous"])
    n_amb = len(amb_ids)
    if n_amb:
        print(
            f"WARNING: {n_amb} candidates have an ambiguous periodic image: an "
            "anchor pair, or the O-H site separation, is ~half a lattice vector "
            "and the alternative image gives a different graph. Their anchor "
            "indices do not define a unique geometry; one image was scored. "
            "A larger surface cell removes this."
        )
        print(f"  Ambiguous candidate_id values: {amb_ids}")
    if SCORE_NEAR_ONLY:
        print(
            f"Far candidates skipped unscored: {n_skipped_far} "
            f"(site separation > {PAIR_DISTANCE_SPLIT:.2f} A; "
            "SCORE_NEAR_ONLY=True, so this table is near-only)"
        )

    print(f"\n=== TOP {TOP_N_PRINT} CANDIDATES ===")
    for rank, row in enumerate(results[:TOP_N_PRINT], start=1):
        print(
            f"{rank:2d} | "
            f"motif={row['motif']:2s} | "
            f"O={row['o_anchor_idx']} | "
            f"H={row['h_anchor_idx']} | "
            f"shared={row['shared_anchor_count']} | "
            f"Eads_pred={row['pred_Eads_eV']:.6f} eV | "
            f"ret_score={fmt_score(row['retention_probability'])} | "
            f"{row['retention_flag']}"
        )

    if USE_RETENTION_MODEL:
        retained_like = [
            r for r in results
            if r["retention_probability"] >= RETENTION_CONSERVATIVE_THRESHOLD
        ]
        highconf_like = [
            r for r in results
            if r["retention_probability"] >= RETENTION_HIGHCONF_THRESHOLD
        ]

        print("\n=== RETENTION-SCORE SUMMARY ===")
        print(f"Candidates with score >= {RETENTION_SCREENING_THRESHOLD:.2f}: {sum(r['retention_probability'] >= RETENTION_SCREENING_THRESHOLD for r in results)}")
        print(f"Candidates with score >= {RETENTION_CONSERVATIVE_THRESHOLD:.2f}: {len(retained_like)}")
        print(f"Candidates with score >= {RETENTION_HIGHCONF_THRESHOLD:.2f}: {len(highconf_like)}")

        print(f"\n=== TOP {TOP_N_PRINT} CANDIDATES WITH RETENTION SCORE >= {RETENTION_CONSERVATIVE_THRESHOLD:.2f} ===")
        for rank, row in enumerate(retained_like[:TOP_N_PRINT], start=1):
            print(
                f"{rank:2d} | "
                f"motif={row['motif']:2s} | "
                f"O={row['o_anchor_idx']} | "
                f"H={row['h_anchor_idx']} | "
                f"Eads_pred={row['pred_Eads_eV']:.6f} eV | "
                f"ret_score={fmt_score(row['retention_probability'])} | "
                f"{row['retention_flag']}"
            )

    # ============================================================
    # SAVE CSV FILES
    # ============================================================

    # Simple final CSV: one exact-site retention probability + one interpretation flag.
    csv_header = [
        "rank",
        "candidate_id",
        "motif",
        "o_type",
        "h_type",
        "o_anchor_idx",
        "h_anchor_idx",
        "o_candidate_representation",
        "h_candidate_representation",
        "shared_anchor_count",
        "distance_class",
        "min_cross_dist_A",
        "mean_cross_dist_A",
        "max_cross_dist_A",
        "site_distance_A",
        "pbc_image_ambiguous",
        "pred_Eads_eV",
        "retention_probability",
        "retention_flag",
        "seen_motif",
        "support_count",
        "support_o_anchor_rule_counts",
        "support_h_anchor_rule_counts",
        "ood_score",
        "ood_reason",
    ]

    # --- full ranked file ---
    OUT_CSV_ALL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV_ALL, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        for rank, row in enumerate(results, start=1):
            writer.writerow([
                rank,
                row["candidate_id"],
                row["motif"],
                row["o_type"],
                row["h_type"],
                str(row["o_anchor_idx"]),
                str(row["h_anchor_idx"]),
                row["o_candidate_representation"],
                row["h_candidate_representation"],
                row["shared_anchor_count"],
                row["distance_class"],
                f"{row['min_cross_dist']:.6f}",
                f"{row['mean_cross_dist']:.6f}",
                f"{row['max_cross_dist']:.6f}",
                f"{row.get('site_distance', np.nan):.6f}",
                row["pbc_image_ambiguous"],
                f"{row['pred_Eads_eV']:.8f}",
                fmt_score(row["retention_probability"]),
                row["retention_flag"],
                row["seen_motif"],
                row["support_count"],
                str(row["support_o_anchor_rule_counts"]),
                str(row["support_h_anchor_rule_counts"]),
                row["ood_score"],
                row["ood_reason"],
            ])

    print(f"\nSaved full ranking to: {OUT_CSV_ALL}")
    settings_path = write_prediction_settings(
        OUT_CSV_ALL,
        domain_label,
        len(results),
        ckpt,
        retention_ckpt,
    )
    print(f"Saved run settings to: {settings_path}")

if __name__ == "__main__":
    main()

