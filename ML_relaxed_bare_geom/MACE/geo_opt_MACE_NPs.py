from ase.io import read, write
from ase.optimize import FIRE
from mace.calculators import mace_mp
import os


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------
model_path = "mace-mh-1.model"
atoms = read("./Ni55_ini.xyz", format="xyz")

atoms.set_cell([22.0, 22.0, 22.0])
atoms.set_pbc([True, True, True])


# ------------------------------------------------------------
# MACE-MH-1 calculator
# ------------------------------------------------------------
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
        append=True,
    )

save_xyz()


# ------------------------------------------------------------
# Geometry optimization
# ------------------------------------------------------------
optimizer = FIRE(
    atoms,
    trajectory="opt_MACE_MH1.traj",
    logfile="opt_MACE_MH1.log",
)

optimizer.attach(save_xyz, interval=1)

optimizer.run(
    fmax=0.02,
    steps=5000,
)


# ------------------------------------------------------------
# Final geometry
# ------------------------------------------------------------
write(
    "Ni55_MACE_opt.xyz",
    atoms,
    format="xyz",
)
