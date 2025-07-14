# Evaluation

## How to read `log.lammps`

- This section gives the end-to-end MD simulation time (including LAMMPS), the number of MPI process, the number of MD steps and the number of atoms.

```
Loop time of 3.02717 on 6 procs for 100 steps with 23040 atoms
```

- This section gives the breakdown of the run time (in seconds) into major categories:
  - Pair = non-bonded force computations (i.e. run time of TensorMD/DP excluding LAMMPS)
  - Neigh = neighbor list construction
  - Comm = inter-processor communication of atoms and their properties
  - Output = output of thermodynamic info and dump files
  - Modify = fixes and computes invoked by fixes
  - Other = all the remaining time

```
MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 2.9445     | 2.9445     | 2.9445     |   0.0 | 97.27
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0.013914   | 0.013914   | 0.013914   |   0.0 |  0.46
Output  | 0.0031512  | 0.0031512  | 0.0031512  |   0.0 |  0.10
Modify  | 0.055007   | 0.055007   | 0.055007   |   0.0 |  1.82
Other   |            | 0.01063    |            |       |  0.35
```

- This is the additional section added by TensorMD, reporting the FLOPs, runtime and FLOP/s of each kernel. The following shows the correspondence between the kernel and the equation in the paper.
  - Encoding phase: `setup`, `fnn->interp` (Eq.4), `Kernel1` (Eq.5, 6, 7)
  - Regression phase: `nn->compute` (Eq.8)
  - Derivation phase: `dEdP` (Eq.9), `V` (Eq.11), `F2` (Eq.12), `Kernel4` (Eq. 10, 12), `forces`

```
--------------------------------------------------------------------------

                               FLOPs Report

--------------------------------------------------------------------------
         alloc:     0.0000 TFlops,       0.0000 secs,       0.0000 TFLOP/s
         setup:     0.0000 TFlops,       0.5626 secs,       0.0000 TFLOP/s
   fnn->interp:     0.1625 TFlops,       0.3508 secs,       0.4632 TFLOP/s
       Kernel1:     0.3747 TFlops,       0.3711 secs,       1.0096 TFLOP/s
   nn->compute:     0.7615 TFlops,       0.3916 secs,       1.9445 TFLOP/s
          dEdP:     0.0327 TFlops,       0.1467 secs,       0.2232 TFLOP/s
             V:     0.3421 TFlops,       0.3360 secs,       1.0183 TFLOP/s
            F2:     0.0057 TFlops,       0.0794 secs,       0.0724 TFLOP/s
       Kernel4:     0.3596 TFlops,       0.5071 secs,       0.7091 TFLOP/s
        forces:     0.0016 TFlops,       0.1826 secs,       0.0088 TFLOP/s
--------------------------------------------------------------------------
       Overall:     2.0404 TFlops,       2.9278 secs,       0.6969 TFLOP/s
--------------------------------------------------------------------------
```

- For more detail information, please refer to [Screen and logfile output](https://docs.lammps.org/Run_output.html).

## Install DP and TensorMD

1. For installing DP(v3.0.0) with LAMMPS, please refer to [DeePMD-kit’s documentation](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html). Set global variable `DP_PATH` to the location of executable `lmp`, i.e. `export DP_PATH=/path/to/lmp`.

2. Follow the instructions in `lammps\src\ML-TENSORMD\README.md` to compile the baseline implementation of TensorMD. Set global variable `TENSORMD_BASELINE_PATH` to the location of executable `lmp`, i.e. `export TENSORMD_BASELINE_PATH=/path/to/lmp`.

3. Follow the instructions in `lammps\src\ML-TENSORMD\README.md` to compile the optimized implementation of TensorMD (recommand to set `-DENABLE_CUDA_WMMA=yes` for cuda). Set global variable `TENSORMD_PATH` to the location of executable `lmp`, i.e. `export TENSORMD_PATH=/path/to/lmp`.


## Figure 8
This figure shows the execution times (excluding LAMMPS) of TensorMD and DP for batching different numbers of W atoms on A100 and ORISE.

1. Run the test.
```bash
cd lammps\examples\tensormd
bash run_compare.sh 
```

2. The result CSV file is `./compare.csv`.

3. The original data of A100 and ORISE are in `lammps\examples\tensormd\W\dp\compare\a100_log`, `lammps\examples\tensormd\W\dp\compare\orise_log`, `lammps\examples\tensormd\W\tensormd\compare\a100_log` and `lammps\examples\tensormd\W\tensormd\compare\orise_log`.

## Figure 9
This figure shows the cumulative speedups of TensorMD over its baseline. The speedups are calculated using the end-to-end MD simulation time (including LAMMPS).

1. Run the test.
```bash
cd lammps\examples\tensormd
bash run_compare.sh 
```

2. The result CSV file is `./speedup.csv`.

3. The original data of A100 and ORISE are in `a100_log/log_baseline.lammps` and `orise_log/log_baseline.lammps` respectively in directory `lammps\examples\tensormd\W\tensormd\speedup` (W), `lammps\examples\tensormd\binary\medium` (MoNi) and `lammps\examples\tensormd\trinary` (AlMgSi).

## Figure 11, 13
These figures show the scaling result of TensorMD on ORISE. The test inputs, script and original data are in directory `lammps\examples\tensormd\scaling`.