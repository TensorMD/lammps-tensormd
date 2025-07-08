#!/bin/bash
export OMP_NUM_THREADS=40
mpirun -np 1 /home/oyyc/lammps/build/lmp -in in.lammps
