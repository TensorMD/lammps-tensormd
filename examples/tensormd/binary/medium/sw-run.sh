bsub -I -J TensorMD -q q_sw_expr -b -n 6 -cgsp 64 -share_size 15520 ../../../../build/lmp -in in.lammps
# bsub -I -J TensorMD -q q_sw_expr -b -n 6 -cgsp 64 -share_size 15520 ../../../../build_baseline/lmp -in in.lammps
