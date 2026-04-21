#!/bin/bash

#SBATCH --job-name=Sn
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

module load cuda-12.4 tbb/latest compiler-rt/2024.2.0 compiler/2024.2.0 mpi/2021.13 mkl/2024.2
export PATH=/share/home/chenxin/software/tensormd/dev/build/release:$PATH

export TENSORMD_INTERP_DELTA=0.01
export TENSORMD_VERBOSE_DEVICE_MAP=true
export TENSORMD_MEMORY_GROWTH_RATE=1.04
export TENSORMD_TIMER=on
export TENSORMD_DEVICE=gpu
export TENSORMD_STRATEGY=Scale

results_dir=results/natoms_vs_speed
if [ ! -d "$results_dir" ]; then
    mkdir -p "$results_dir"
fi

export OMP_THREADS=6

precision="fp32"

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 20 \
    -var ny 20 \
    -var nz 10 \
    -log $results_dir/log.16000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 20 \
    -var ny 20 \
    -var nz 20 \
    -log $results_dir/log.32000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 40 \
    -var ny 20 \
    -var nz 20 \
    -log $results_dir/log.64000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 40 \
    -var ny 40 \
    -var nz 20 \
    -log $results_dir/log.128000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 40 \
    -var ny 40 \
    -var nz 40 \
    -log $results_dir/log.256000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 80 \
    -var ny 40 \
    -var nz 40 \
    -log $results_dir/log.512000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 80 \
    -var ny 80 \
    -var nz 40 \
    -log $results_dir/log.1024000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 80 \
    -var ny 80 \
    -var nz 80 \
    -log $results_dir/log.2048000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 160 \
    -var ny 80 \
    -var nz 80 \
    -log $results_dir/log.4096000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 160 \
    -var ny 160 \
    -var nz 80 \
    -log $results_dir/log.8192000.lammps

mpiexec -np 1 lmp_tensormd -pk omp $OMP_THREADS neigh yes \
    -in in.lammps \
    -var model models/Sn_k32_m4_rc6.0_fp32.npz \
    -var phase beta \
    -var nx 160 \
    -var ny 160 \
    -var nz 160 \
    -log $results_dir/log.163840000.lammps
