# Sensitivity of hybrid Ni–O/Ni–H adsorption-site assignment (M6 datasets)

## Input files

- NP: `dataset_NPs_M6_energy_model.extxyz` — 1,865 frames, containing Ni25/30/35/40/45/47/50.
- Slabs: `dataset_SLABs_M6_energy_model.extxyz` — 216 frames, Ni(111) (90) and Ni(100) (126).
- Total: 2,081 frames, including all records in the supplied files.

## Implemented assignment

- Nearest Ni contacts to O and adsorbed H, with minimum-image distances for periodic triclinic slab cells.
- Standard cutoffs: O = 2.20 Å, H = 2.00 Å. Ordinary T, B, H sites correspond to 1, 2, 3 Ni contacts; at most the three nearest ordinary contacts are retained.
- Extended F cutoffs: O = 2.30 Å, H = 2.20 Å; exactly 4 contacts within an extended cutoff and at least 2 within the corresponding short cutoff, otherwise ordinary assignment.
- The hydrogen closest to O is the OH hydrogen; the other is adsorbed H. One NP frame violates the assumed OH bond, flagged below.
- Motif/anchor stability defined by exact equality with assignments at the corresponding baseline thresholds, within each fixed frame; no models are retrained.

The implemented hybrid rule exactly reproduces both original motif-count tables supplied by the user: at standard H cutoff 1.90 and 2.00 Å with O = 2.20 Å, all 13 motif populations agree.

## Standard-cutoff grid

- Standard Ni–O = 2.10, 2.15, 2.20, 2.25, 2.30 Å.
- Standard Ni–H = 1.90, 1.95, 2.00, 2.05, 2.10 Å.
- Extended cutoffs held fixed (O = 2.30, H = 2.20 Å).
- 25 combinations, evaluated independently for NPs and slabs.

Key results:

| Standard O/H cutoffs (Å) | NP altered motifs (n/1865) | NP % | Slab altered motifs (n/216) | Slab % |
|:--|--:|--:|--:|--:|
| 2.20 / 1.90 | 29 | 1.55 | 0 | 0.00 |
| 2.20 / 1.95 | 14 | 0.75 | 0 | 0.00 |
| 2.20 / 2.05 | 3 | 0.16 | 0 | 0.00 |
| 2.20 / 2.10 | 4 | 0.21 | 0 | 0.00 |
| 2.15 / 2.00 | 9 | 0.48 | 0 | 0.00 |
| 2.25 / 2.00 | 2 | 0.11 | 0 | 0.00 |
| 2.10 / 2.00 | 60 | 3.22 | 34 | 15.74 |
| 2.10 / 1.90 | 87 | 4.66 | 34 | 15.74 |

At any point of the ±0.05 Å by ±0.05 Å grid about the baseline, no more than 23/1865 NP motifs (1.23%) change and zero slab motifs change. Across the full ±0.10 Å grid, up to 87/1865 NP motifs change (4.66%), and up to 34/216 slab motifs change (15.74%). All 34 slab changes are Ni(100), and only occur when O short cutoff is 2.10 Å. Those slab shifts mainly involve loss or relabeling of nominal fourfold sites, as the requirement of at least two short O contacts is no longer met. For every standard-cutoff grid cell, exact anchor-set changes coincided with motif-label changes, and there were no additional configurations with unchanged motif but changed anchor identities.

At O = 2.20 Å, changing H from 2.00 to 1.90 changes exactly 29 NP motifs: 4 BH→BB, 22 HH→HB, 3 TH→TB. Slab labels are all unchanged.

## Extended fourfold-cutoff grid

- Standard O and H cutoffs fixed at 2.20 and 2.00 Å.
- Extended O = 2.20, 2.25, 2.30, 2.35, 2.40 Å.
- Extended H = 2.10, 2.15, 2.20, 2.25, 2.30 Å.
- At every point within ±0.05 Å of the extended baseline (O = 2.30, H = 2.20 Å), **all 2,081 assignments are unchanged**.
- Across the ±0.10 Å extended grid, the most changed motifs occur at extended O = 2.20 and extended H = 2.10 Å: 3/1865 NPs (0.16%) and 6/216 slabs (2.78%).

## Motif-energy interpretation

For the NP DFT training set, mean adsorption energies grouped by assigned motifs change by at most approximately 11 meV for BH, 2.3 meV for HH and 1.1 meV for TH across the entire standard cutoff grid. These numbers describe only the grouping of *already calculated DFT reference energies*, not the GNN's errors or its predictions for much larger nanoparticles. No claim about the stability of GCN-versus-predicted-energy slopes is established by these files alone; that would require access to the relevant bare structures/GNN predictions or descriptors.

## Reproducibility and limitations

- The scan directory is stored at repository level as
  `ni_cutoff_sensitivity_scan/`. Its scripts resolve the two input datasets
  from the sibling `datasets/` directory and can be launched from any working
  directory.
- `cutoff_scan.py` generates the 25-point standard-cutoff scan and all per-configuration reassignment records.
- `ni_cutoff_extended.py` generates the 25-point extended-fourfold scan.
- ASE is not required. These scripts parse `extxyz` directly, and apply lattice translations when computing slab distances.
- A selected cutoff range establishes robustness of motif/anchor *assignments* on the supplied DFT-relaxed training structures. This does not alone validate whether the cutoff gives a physically correct site assignment for every distorted geometry or whether the training/retention metrics would be invariant after retraining.
- Dataset counts include possible duplicate configuration labels (not deduplicated within this study).
