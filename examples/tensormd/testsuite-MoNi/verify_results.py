import numpy as np
import json
import pandas as pd
from pathlib import Path
from ase.io.lammpsrun import read_lammps_dump

cases = [
    "c1_pot-MoNi_Mo",
    "c2_pot-MoNi_Ni",
    "c3_pot-MoNi_MoNi",
]


def load_expected(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_calculated(filepath):
    try:
        atoms = read_lammps_dump(str(filepath), specorder=["Mo", "Ni"])
    except Exception:
        print(f"Failed to read LAMMPS dump file: {filepath}")
        return None
    energy = atoms.arrays['c_2'].sum()
    forces = atoms.get_forces()
    vol = atoms.get_volume()
    pxx = atoms.arrays['c_1[1]'].sum() / -vol
    pyy = atoms.arrays['c_1[2]'].sum() / -vol
    pzz = atoms.arrays['c_1[3]'].sum() / -vol
    pxy = atoms.arrays['c_1[4]'].sum() / -vol
    pxz = atoms.arrays['c_1[5]'].sum() / -vol
    pyz = atoms.arrays['c_1[6]'].sum() / -vol
    return {
        "energy": energy,
        "forces": forces,
        "pressures": np.array([pxx, pyy, pzz, pyz, pxz, pxy])
    }


for actfn in ("tanh", "silu", "silu32", "softplus"):
    results = {
        "case": [], "actfn": [], "dE": [], "dF(max)": [], "dF(min)": [], "dF(MAE)": [],
        "dP(max)": [], "dP(min)": [], "dP(MAE)": [],
    }
    for run_case in cases:
        case_dir = Path(run_case) / actfn
        results_file = case_dir / f"results"
        reference_file = case_dir / f"reference.json"
        if not (results_file.exists() and reference_file.exists()):
            continue

        expected = load_expected(reference_file)
        calculated = load_calculated(results_file)
        if calculated is None:
            continue

        dE = expected['energy'] - calculated['energy']
        dF = np.abs(expected['forces'] - calculated['forces'])
        dP = np.abs(expected['pressures'] - calculated['pressures'])

        results["case"].append(str(run_case))
        results["actfn"].append(f"{actfn:>8s}")
        results["dE"].append(f"{dE: 12.5e}")
        results["dF(MAE)"].append(f"{dF.mean(): 12.5e}")
        results["dF(max)"].append(f"{dF.max(): 12.5e}")
        results["dF(min)"].append(f"{dF.min(): 12.5e}")
        results["dP(MAE)"].append(f"{dP.mean(): 12.5e}")
        results["dP(max)"].append(f"{dP.max(): 12.5e}")
        results["dP(min)"].append(f"{dP.min(): 12.5e}")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
