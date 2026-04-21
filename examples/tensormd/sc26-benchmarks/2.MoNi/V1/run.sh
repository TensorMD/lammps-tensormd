#!/bin/bash

#SBATCH --job-name=MoNi
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

module purge 
module load tensormat/tensormd/gpu/v2025.04 

OMP_THREADS=6

# The largest run
mpiexec -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model ../models/MoNi_V1_m3k64_fp32.npz \
    -var phase beta \
    -var nx 36 \
    -var ny 40 \
    -var nz 40
