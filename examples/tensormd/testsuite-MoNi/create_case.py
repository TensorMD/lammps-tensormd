import numpy as np
import json
from ase.build import bulk
from ase.units import GPa
from ase.io import read
from ase import Atoms
from pathlib import Path
from ase.io.lammpsdata import write_lammps_data
from tensoralloy.calculator.ase_calculator import TensorAlloyCalculator

np.random.seed(42)

calc = {
    "softplus": TensorAlloyCalculator("MoNi_V2_m5k64mlp_softplus_fp32.pt", active_task_id="MoNi"),
    "tanh": TensorAlloyCalculator("MoNi_V2_m3k128mlp_tanh_fp64.pt", active_task_id="MoNi"),
    "silu": TensorAlloyCalculator("MoNi_V2_m5k64mlp_silu_fp64.pt", active_task_id="MoNi")
}

model_path = {
    "softplus": "../../../../potentials/tensormd/MoNi_V2_m5k64mlp_softplus_fp32.npz",
    "tanh": "../../../../potentials/tensormd/MoNi_V2_m3k128mlp_tanh_fp64.npz",
    "silu": "../../../../potentials/tensormd/MoNi_V2_m5k64mlp_silu_fp64.npz"
}

result_file = "results"


def write_reference_data(struct: Atoms, filepath: str, calculator):
    struct.calc = calculator
    energy = struct.get_potential_energy()
    forces = struct.get_forces()
    stress = struct.get_stress(voigt=True)
    energies = struct.get_potential_energies()
    pressures = stress / GPa * -10000
    with open(filepath, 'w') as f:
        json.dump({
            "energy": energy,
            "forces": forces.tolist(),
            "pressures": pressures.tolist(),
            "energies": energies.tolist()
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
pair_coeff      * * {model_path} Mo Ni
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
    for actfn in ["softplus", "tanh", "silu"]:
        case_subdir = case_dir / f"{actfn}"
        case_subdir.mkdir(parents=True, exist_ok=True)
        obj = struct.copy()
        write_lammps_data(str(case_subdir / "in.data"), obj, units="metal", atom_style="atomic", specorder=["Mo", "Ni"])
        write_reference_data(obj, str(case_subdir / f"reference.json"), calc[actfn])
        write_lammps_in(str(case_subdir / f"in.lammps"))


c1 = Path("c1_pot-MoNi_Mo")
atoms = bulk("Mo", cubic=True)
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.03
create_case(atoms, c1)

c2 = Path("c2_pot-MoNi_Ni")
atoms = bulk("Ni", cubic=True)
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.03
create_case(atoms, c2)

c3 = Path("c3_pot-MoNi_MoNi")
atoms = read("Ni6Mo2.poscar")
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.01
create_case(atoms, c3)
