#!/bin/bash

#SBATCH --job-name=AlCuMg
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

module purge 
module load tensormat/tensormd/gpu/v2025.04 

OMP_THREADS=6

model=$(realpath ../models/AlCuMg_V1_m4k64_fp32.npz)

mpirun -np 1 lmp_tensormd_gpu -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model $model \
    -var nx 20
