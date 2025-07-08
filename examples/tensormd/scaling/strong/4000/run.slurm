#!/bin/bash
#SBATCH --partition=july30
#SBATCH --time=02:00:00
#SBATCH --nodes=4000
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4
#SBATCH --exclude=j01r3n13,j02r4n16,k05r3n09,k06r2n01,i11r2n03

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Two sockets per node
SOCKETS_PER_NODE=1

# Total number of processes
nprocs=$(($SLURM_JOB_NUM_NODES*$SLURM_NTASKS_PER_NODE))

# Processes per socket
pps="1"
if [ "$SLURM_NTASKS_PER_NODE" != "1" ]; then
    pps=$(($SLURM_NTASKS_PER_NODE/$SOCKETS_PER_NODE))
fi

# mpirun -np $nprocs --map-by ppr:$pps:socket:PE=$OMP_NUM_THREADS --bind-to core \
#    /public/home/aicao/oyyc/lammps/build/lmp \
#    -pk omp $OMP_NUM_THREADS neigh yes \
#    -in in.lammps -log log.lammps

mpirun -np $nprocs /public/home/aicao/oyyc/lammps/build/lmp \
    -pk omp $OMP_NUM_THREADS neigh yes \
    -in in.lammps -log log.lammps
