import numpy as np
import json
from ase.build import bulk
from ase.units import GPa
from ase import Atoms
from pathlib import Path
from ase.io.lammpsdata import write_lammps_data

np.random.seed(42)

try:
    from tensoralloy.calculator import TensorAlloyCalculator
    calc = TensorAlloyCalculator("AlCuMg_v1_m3k64softplus_fp64.pb")
    model_path = "../../../../potentials/tensormd/AlCuMg_V1_fp64.npz"
    version = 1
except ImportError:
    from tensoralloy.calculator.ase_calculator import TensorAlloyCalculator
    calc = TensorAlloyCalculator("AlCuMg_v2_m3k128nnp256_fp64.pt", active_task_id="AlMgSi")
    model_path = "../../../../potentials/tensormd/AlCuMg_V2mlp_fp64.npz"
    version = 2
except Exception:
    raise ImportError("TensorAlloy is not installed.")


def write_reference_data(struct: Atoms, filepath: str):
    struct.calc = calc
    energy = struct.get_potential_energy()
    forces = struct.get_forces()
    stress = struct.get_stress(voigt=True)
    if version == 1:
        energies = np.zeros(len(struct))
    else:
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
pair_coeff      * * {model_path} Al Cu Mg
neighbor        2.0 bin
neigh_modify	delay 0 every 1 check yes
fix             1 all nve
thermo_style    custom etotal vol pxx pyy pzz
timestep        0.0001
compute         1 all stress/atom NULL
compute         2 all pe/atom
dump            1 all custom 1 results id type c_2 x y z fx fy fz c_1[*]
dump_modify     1 sort id format float "% 23.15e"
run             0
""")


def create_case(struct: Atoms, case_dir: Path):
    case_dir.mkdir(parents=True, exist_ok=True)
    write_lammps_data(str(case_dir / "in.data"), struct, units="metal", atom_style="atomic", specorder=["Al", "Cu", "Mg"])
    write_reference_data(struct, str(case_dir / f"reference.{version}.json"))
    write_lammps_in(str(case_dir / f"in.{version}.lammps"))


atoms = bulk("Al", cubic=True, a=4.05)
c1 = Path("c1_pot-AlCuMg_Al")
create_case(atoms, c1)

c2 = Path("c2_pot-AlCuMg_Al")
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.01
create_case(atoms, c2)

c3 = Path("c3_pot-AlCuMg_Cu")
atoms = bulk("Cu", cubic=True, a=3.61)
create_case(atoms, c3)

c4 = Path("c4_pot-AlCuMg_Cu")
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.01
create_case(atoms, c4)

c5 = Path("c5_pot-AlCuMg_Mg")
atoms = bulk("Mg", a=3.21, c=5.21)
create_case(atoms, c5)

c6 = Path("c6_pot-AlCuMg_Mg")
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.01
create_case(atoms, c6)

c7 = Path("c7_pot-AlCuMg_AlCu")
atoms = bulk('Al', cubic=True, a=4.05) * [2, 2, 2]
atoms.positions += np.random.randn(*atoms.positions.shape) * 0.01
atoms.set_chemical_symbols(['Al'] * 16 + ['Cu'] * 16)
create_case(atoms, c7)

c8 = Path("c8_pot-AlCuMg_AlMg")
atoms = bulk('Al', cubic=True, a=4.05) * [2, 2, 2]
atoms.positions += np.random.randn(*atoms.positions.shape) * 0
atoms.set_chemical_symbols(['Al'] * 16 + ['Mg'] * 16)
create_case(atoms, c8)

c9 = Path("c9_pot-AlCuMg_CuMg")
atoms = bulk('Cu', cubic=True, a=3.61) * [2, 2, 2]
atoms.positions += np.random.randn(*atoms.positions.shape) * 0
atoms.set_chemical_symbols(['Cu'] * 16 + ['Mg'] * 16)
create_case(atoms, c9)

c10 = Path("c10_pot-AlCuMg_AlCuMg")
atoms = bulk('Al', cubic=True, a=4.05) * [2, 2, 2]
atoms.positions += np.random.randn(*atoms.positions.shape) * 0
atoms.set_chemical_symbols(['Al'] * 16 + ['Cu'] * 8 + ['Mg'] * 8)
create_case(atoms, c10)

c11 = Path("c11_pot-AlCuMg_AlCuMg")
atoms = bulk('Al', cubic=True, a=4.05)
atoms.set_chemical_symbols(['Al', 'Cu', 'Mg', 'Al'])
create_case(atoms, c11)
