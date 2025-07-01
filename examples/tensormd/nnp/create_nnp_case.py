import numpy as np
import yaml
import datetime
from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
from ase.units import fs
from ase.data import atomic_numbers
from ase.md import VelocityVerlet
from tensoralloy.calculator import TensorAlloyCalculator

np.random.seed(10)

atoms = bulk("Mo", crystalstructure="bcc", cubic=True) * [2, 2, 2]
atoms.set_chemical_symbols(["Mo"] * 8 + ["Ni"] * 4 + ["Mo"] * 4)
atoms.positions += np.random.randn(16, 3) * 0.04

write_lammps_data("in.data", atoms, specorder=["Mo", "Ni"])
atoms = read_lammps_data(
    "in.data", 
    Z_of_type={1: atomic_numbers["Mo"], 2: atomic_numbers["Ni"]},
    style="atomic")

calc = TensorAlloyCalculator("MoNi.pb")
atoms.calc = calc
calc.calculate(atoms, properties=("energy", "forces", "stress"))

E1 = calc.get_potential_energy(atoms)
F1 = calc.get_forces(atoms)
v1 = (-calc.get_stress(atoms) * atoms.get_volume())[[0, 1, 2, 5, 4, 3]]

dyn = VelocityVerlet(atoms, dt=0.1 * fs,
                     trajectory='md.traj', logfile='md.log')
dyn.run(4)

E2 = calc.get_potential_energy(atoms)
F2 = calc.get_forces(atoms)
v2 = (-calc.get_stress(atoms) * atoms.get_volume())[[0, 1, 2, 5, 4, 3]]

data = {
    "lammps_version": "Aug 2023",
    "date_generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "epsilon": 2e-12,
    "skip_tests": "",
    "prerequisites": "pair tensormd",
    "pre_commands": "",
    "post_commands": "",
    "input_file": "in.tensormd-tdnp",
    "pair_style": "tensormd",
    "pair_coeff": "* * ../../../potentials/tensormd/MoNi_fp64.npz Mo Ni",
    "extract": "", 
    "natoms": 16,
    "init_vdwl": float(E1),
    "init_coul": 0,
    "init_stress": " ".join("{: 23.16e}".format(v) for v in v1),
    "init_forces": "\n".join([f"{i + 1:2d} {fx: 23.16e} {fy: 23.16e} {fz: 23.16e}" for i, (fx, fy, fz) in enumerate(F1)]),  
    "run_vdwl": float(E2),
    "run_coul": 0,
    "run_stress": " ".join("{: 23.16e}".format(v) for v in v2),
    "run_forces": "\n".join([f"{i + 1:2d} {fx: 23.16e} {fy: 23.16e} {fz: 23.16e}" for i, (fx, fy, fz) in enumerate(F2)]),
}

with open("manybody-pair-tensormd-nnp.yaml", "w") as fp:
    yaml.dump(data, fp, 
              indent=2, 
              default_style='|', 
              sort_keys=False,
              explicit_end=True, 
              explicit_start=True)
