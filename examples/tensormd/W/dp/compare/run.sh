#!/bin/bash
export OMP_NUM_THREADS=40
# export TF_INTRA_OP_PARALLELISM_THREADS=1
# export TF_INTER_OP_PARALLELISM_THREADS=40
mpirun -np 1 lmp -in in.lammps 
