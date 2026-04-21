#!/bin/bash

lammps_root=/share/home/chenxin/software/tensormd/dev

export PATH=$HOME/software/tensormd/dev/build/debug:$PATH

export TENSORMD_DEVICE=cpu
export TENSORMD_INTERP_DELTA=0.0001
export TENSORMD_STRATEGY=U_abcd
export TENSORMD_SVD_NBASIS=64
export TENSORMD_SVD_PATH=CACHE

# export TENSORMD_DUMP_TENSORS=P,G
export TENSORMD_GPU_SUM_FORCES=new

export OMP_NUM_THREADS=1

lmp_tensormd -in in.lammps
