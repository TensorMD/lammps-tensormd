import numpy as np
import json
import pandas as pd
from pathlib import Path
from ase.io.lammpsrun import read_lammps_dump

cases = [
    "c1_pot-AlCuMg_Al",
    "c2_pot-AlCuMg_Al",
    "c3_pot-AlCuMg_Cu",
    "c4_pot-AlCuMg_Cu",
    "c5_pot-AlCuMg_Mg",
    "c6_pot-AlCuMg_Mg",
    "c7_pot-AlCuMg_AlCu",
    "c8_pot-AlCuMg_AlMg",
    "c9_pot-AlCuMg_CuMg",
    "c10_pot-AlCuMg_AlCuMg",
    "c11_pot-AlCuMg_AlCuMg",
]


def load_expected(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_calculated(filepath):
    try:
        atoms = read_lammps_dump(str(filepath), specorder=["Al", "Cu", "Mg"])
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


for version in (1, 2):
    results = {
        "case": [], "dE": [], "dF(max)": [], "dF(min)": [], "dF(MAE)": [], "dF(RMSE)": [],
        "dP(max)": [], "dP(min)": [], "dP(MAE)": [], "dP(RMSE)": []
    }
    for run_case in cases:
        case_dir = Path(run_case)
        results_file = case_dir / f"results.{version}"
        reference_file = case_dir / f"reference.{version}.json"
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
        results["dE"].append(f"{dE: 12.5e}")
        results["dF(MAE)"].append(f"{dF.mean(): 12.5e}")
        results["dF(max)"].append(f"{dF.max(): 12.5e}")
        results["dF(min)"].append(f"{dF.min(): 12.5e}")
        results["dF(RMSE)"].append(f"{np.sqrt((dF**2).mean()): 12.5e}")
        results["dP(MAE)"].append(f"{dP.mean(): 12.5e}")
        results["dP(max)"].append(f"{dP.max(): 12.5e}")
        results["dP(min)"].append(f"{dP.min(): 12.5e}")
        results["dP(RMSE)"].append(f"{np.sqrt((dP**2).mean()): 12.5e}")

    df = pd.DataFrame(results)
    sep = "--------------------------------------------------"
    print(f"{sep} TensorMDV{version} {sep}")
    print(df.to_string(index=False))
