#!/bin/bash

lammps_root=/share/home/chenxin/software/tensormd/dev

export PATH=$HOME/software/tensormd/dev/build/release:$PATH

export TENSORMD_DEVICE=gpu
export TENSORMD_INTERP_DELTA=0.001
export TENSORMD_STRATEGY=All_Fused
export TENSORMD_INTERP_ALGO=cubic
export TENSORMD_PGW_K_CONTINUOUS=true
export TENSORMD_USE_FP32_ATLAS=false

export TENSORMD_DUMP_TENSORS=Fr,Fa,W,U,V,G,P
export OMP_NUM_THREADS=1

if [ "${TENSORMD_DEVICE}" = "gpu" ]; then
  if [ "$1" == "gdb" ] ; then
    cuda-gdb lmp_tensormd
  else
    lmp_tensormd -in in.2.lammps -log log.lammps -var LAMMPS_ROOT ${lammps_root}
  fi
else
  lmp_tensormd -in in.2.lammps -log log.lammps -var LAMMPS_ROOT ${lammps_root}
fi
