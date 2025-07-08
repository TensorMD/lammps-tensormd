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

## Figure 9
This figure shows the cumulative speedups of TensorMD over its baseline. The speedups are calculated using the end-to-end MD simulation time (including LAMMPS).

1. Run `baseline`
    1. Follow the instructions in `lammps\src\ML-TENSORMD\README.md` to compile the baseline implementation of TensorMD.
    2. Add parameter `cmax default` to the last of section `pair_coeff` in file `lammps\examples\tensormd\W\speedup\in.lammps` (W), `lammps\examples\tensormd\binary\medium\in.lammps` (MoNi) and `lammps\examples\tensormd\trinary\in.lammps` (AlMgSi), i.e. ```pair_coeff	  * * ../W_k32_snapshot.npz W interp 0.001 cmax default```.
    3. Run the tests for W, MoNi and AlMgSi with TensorMD baseline implementation.
    4. The original data of A100 and ORISE are in `a100_log/log_baseline.lammps` and `orise_log/log_baseline.lammps` respectively in directory `lammps\examples\tensormd\W\speedup` (W), `lammps\examples\tensormd\binary\medium` (MoNi) and `lammps\examples\tensormd\trinary` (AlMgSi).
2. Run `+ Flexible neighbor-list padding`
    1. Remove parameter `cmax default`.
    2. Run the tests for W, MoNi and AlMgSi with TensorMD baseline implementation.
    3. The original data of A100 and ORISE are in `a100_log/log_baseline_cmax.lammps` and `orise_log/log_baseline_cmax.lammps` respectively in directory `lammps\examples\tensormd\W\speedup` (W), `lammps\examples\tensormd\binary\medium` (MoNi) and `lammps\examples\tensormd\trinary` (AlMgSi).
3. Run `+ GEMM efficiency enhancement` and `+ Forward-backward strength reduction`
    1. Follow the instructions in `lammps\src\ML-TENSORMD\README.md` to compile the optimized implementation of TensorMD (recommand to set `-DENABLE_CUDA_WMMA=yes` for cuda). 
    2. Run the tests for W, MoNi and AlMgSi with TensorMD optimized implementation.
    3. The original data of A100 and ORISE are in `a100_log/log_release.lammps` and `orise_log/log_release.lammps` respectively in directory `lammps\examples\tensormd\W\speedup` (W), `lammps\examples\tensormd\binary\medium` (MoNi) and `lammps\examples\tensormd\trinary` (AlMgSi).
    4. The decrease of end-to-end MD simulation time in kernel `fnn->interp` is attributed to `Forward-backward strength reduction`. The decrease of end-to-end MD simulation time in other kernels is attributed to `GEMM efficiency enhancement`.

## Figure 10, 12
These figures show the scaling result of TensorMD on Sunway. We release the test inputs, script and original data in directory `lammps\examples\tensormd\scaling`, but accessing to Sunway is not permitted.