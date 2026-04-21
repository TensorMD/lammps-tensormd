#!/bin/bash

#SBATCH --job-name=Sn
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

module purge 
module load tensormat/tensormd/gpu/v2025.04 

OMP_THREADS=6

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase delta \
    -var alat 3.4 \
    -var nx 50 \
    -var ny 50 \
    -var nz 50 \
    -log pressure/log.alat34.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase delta \
    -var alat 3.3 \
    -var nx 50 \
    -var ny 50 \
    -var nz 50 \
    -log pressure/log.alat33.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase delta \
    -var alat 3.2 \
    -var nx 50 \
    -var ny 50 \
    -var nz 50 \
    -log pressure/log.alat32.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase delta \
    -var alat 3.1 \
    -var nx 50 \
    -var ny 50 \
    -var nz 50 \
    -log pressure/log.alat31.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase delta \
    -var alat 3.0 \
    -var nx 50 \
    -var ny 50 \
    -var nz 50 \
    -log pressure/log.alat30.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase beta \
    -var nx 20 \
    -var ny 20 \
    -var nz 10 \
    -log ambient/log.16000.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase beta \
    -var nx 20 \
    -var ny 20 \
    -var nz 20 \
    -log ambient/log.32000.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase beta \
    -var nx 40 \
    -var ny 20 \
    -var nz 20 \
    -log ambient/log.64000.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase beta \
    -var nx 40 \
    -var ny 40 \
    -var nz 20 \
    -log ambient/log.128000.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase beta \
    -var nx 40 \
    -var ny 40 \
    -var nz 40 \
    -log ambient/log.256000.lammps

mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase beta \
    -var nx 80 \
    -var ny 40 \
    -var nz 40 \
    -log ambient/log.512000.lammps

# The largest run
mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/Sn_V1_m4k64_fp32.npz \
    -var phase beta \
    -var nx 64 \
    -var ny 64 \
    -var nz 64

