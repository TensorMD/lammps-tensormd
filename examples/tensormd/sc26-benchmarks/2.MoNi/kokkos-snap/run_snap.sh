#!/bin/bash

#SBATCH --job-name=MoNi
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

module load cuda-12.4 tbb/latest compiler-rt/2024.2.0 compiler/2024.2.0 mpi/2021.13 mkl/2024.2
export PATH=/share/home/chenxin/software/tensormd/dev/build/kokkos:$PATH

mpiexec -np 1 lmp_kokkos -in snap.lammps -k on g 1 -sf kk -var nx 160
