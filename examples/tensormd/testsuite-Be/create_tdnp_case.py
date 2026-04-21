import numpy as np
import json
from pathlib import Path
from ase import Atoms
from ase.units import GPa, kB
from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data
from tensoralloy.calculator import TensorAlloyCalculator
from tensoralloy.atoms_utils import set_electron_temperature, get_electron_temperature

np.random.seed(42)

calc = TensorAlloyCalculator("Be_V1_tdnp_v5b7_fp64.pb")
model_path = "../../../../potentials/tensormd/Be_V1_tdnp_v5b7_fp64.npz"
result_file = "results"
kelvin = 2000


def write_reference_data(struct: Atoms, filepath: str):
    struct.calc = calc
    calc.calculate(struct, properties=["energy", "forces", "stress", "free_energy"])
    energy = struct.get_potential_energy()
    forces = struct.get_forces()
    stress = struct.get_stress(voigt=True)
    electron_entropy = calc.get_electron_entropy(struct)
    free_energy = calc.get_free_energy(struct)
    electron_temperature = get_electron_temperature(struct)
    pressures = stress / GPa * -10000
    with open(filepath, 'w') as f:
        json.dump({
            "energy": energy,
            "free_energy": free_energy,
            "electron_entropy": electron_entropy,
            "electron_temperature": electron_temperature,
            "forces": forces.tolist(),
            "pressures": pressures.tolist(),
        }, f, indent=2)


def write_lammps_in(filepath):
    with open(filepath, 'w') as f:
        f.write(f"""
units           metal
boundary        p p p
atom_style      atomic
newton          on
read_data       in.data
pair_style      tensormd
pair_coeff      * * {model_path} Be etemp {kelvin}
neighbor        2.0 bin
neigh_modify	delay 0 every 1 check yes
fix             1 all nve
thermo_style    custom etotal vol pxx pyy pzz
timestep        0.0001
compute         1 all stress/atom NULL
compute         2 all pe/atom
dump            1 all custom 1 {result_file} id type c_2 x y z fx fy fz c_1[*]
dump_modify     1 sort id format float "% 23.15e"
run             0
""")


def create_case(struct: Atoms, case_dir: Path):
    case_dir.mkdir(parents=True, exist_ok=True)
    obj = struct.copy()
    write_lammps_data(str(case_dir / "in.data"), obj, units="metal", atom_style="atomic", specorder=["Be"])
    write_reference_data(obj, str(case_dir / f"reference.json"))
    write_lammps_in(str(case_dir / f"in.lammps"))


c1 = Path("c1_pot-Be_scalar_etemp")
atoms = bulk("Be", crystalstructure="bcc", a=2.10, cubic=True)
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.03
set_electron_temperature(atoms, kelvin * kB)
create_case(atoms, c1)
