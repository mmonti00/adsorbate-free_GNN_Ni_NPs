from ase.io import read, write
from ase.optimize import FIRE
from mace.calculators import mace_mp
import numpy as np
import os


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------
model_path = "mace-mh-1.model"
atoms = read("./Ni100_4x4_ini.vasp", format="vasp")

# Keep 3D periodicity, as in the DFT slab calculation.
# The vacuum along z prevents interaction between periodic images.
atoms.set_pbc([True, True, True])

print("Cell:")
print(atoms.cell)
print("PBC:", atoms.pbc)
print("Number of atoms:", len(atoms))


# ============================================================
# MACE-MH-1 Calculator
# ============================================================
device = "cuda"

atoms.calc = mace_mp(
    model=model_path,
    device=device,
    default_dtype="float64",
    head="omat_pbe",
)


# ------------------------------------------------------------
# Trajectory output
# ------------------------------------------------------------
xyzfile = "relaxation_MACE_MH1.extxyz"

if os.path.exists(xyzfile):
    os.remove(xyzfile)


def save_xyz():
    write(
        xyzfile,
        atoms,
        format="extxyz",
        append=True
    )


save_xyz()


# ------------------------------------------------------------
# Geometry optimization
# ------------------------------------------------------------
optimizer = FIRE(
    atoms,
    trajectory="opt_MACE_MH1.traj",
    logfile="opt_MACE_MH1.log"
)

optimizer.attach(save_xyz, interval=1)

optimizer.run(
    fmax=0.02,
    steps=5000
)


# ------------------------------------------------------------
# Final geometry
# ------------------------------------------------------------
write(
    "Ni100_4x4_MACE_opt.extxyz",
    atoms,
    format="extxyz"
)

write(
    "Ni100_4x4_MACE_opt.vasp",
    atoms,
    format="vasp",
    direct=True,
    vasp5=True
)
