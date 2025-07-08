#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1

module load compiler/llvm-openmp/openmp
module unload compiler/rocm/dtk/22.10.1
module load compiler/rocm/dtk/24.04.3

export OMP_NUM_THREADS=6
mpirun /public/home/liuying/lammps/build/lmp -in in.lammps -log log.lammps
