#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1

module load apps/anaconda3/2022.10
module unload compiler/rocm/dtk/22.10.1
module load compiler/rocm/dtk/22.10

export OMP_NUM_THREADS=6
# export DP_INTER_OP_PARALLELISM_THREADS=2
# export DP_INTRA_OP_PARALLELISM_THREADS=2
mpirun /public/home/liuying/lammps/build_tf/lmp -in in.lammps -log log.lammps
