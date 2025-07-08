#!/bin/bash
BIN="/home/export/online1/mdt00/shisuan/swustcfd/oyyc/sc24/lmp32_big"
bsub -J k32_1_1 -o output -q q_ustc -b -n 6000 -cgsp 64 -share_size 15500 $BIN -in in.lammps

