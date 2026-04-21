#!/bin/bash

#SBATCH --job-name=HEA
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

module load cuda-12.4 tbb/latest compiler-rt/2024.2.0 compiler/2024.2.0 mpi/2021.13 mkl/2024.2
export PATH=/share/home/chenxin/software/tensormd/dev/build/release:$PATH

export TENSORMD_INTERP_DELTA=0.01
export TENSORMD_VERBOSE_DEVICE_MAP=true
export TENSORMD_MEMORY_GROWTH_RATE=1.05
export TENSORMD_TIMER=on
export TENSORMD_DEVICE=gpu
export TENSORMD_STRATEGY=Production
export TENSORMD_USE_HYBRID_KERNEL=1

results_dir=results
if [ ! -d "$results_dir" ]; then
    mkdir -v "$results_dir"
fi

for k in 32 64; do
    for m in 4 6; do
        model=$(realpath models/SSE_k${k}_m${m}_rc6.0_fp32.npz)
        if [ ! -f "$model" ]; then
            echo "Model file not found: $model"
            continue
        fi
        for MPS in 1 8; do
            taskdir="${results_dir}/SSE_k${k}_m${m}_MPS${MPS}_hybrid1"
            if [ ! -d "$taskdir" ]; then
                mkdir -v -p "$taskdir"
            fi
            OMP_THREADS=6
            nvidia-cuda-mps-control -d > /dev/null 2>&1
            mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
                -in in.lammps \
                -var model $model \
                -var nx 10 \
                -var ny 10 \
                -var nz 10 \
                -log "${taskdir}/log.lammps"
            echo quit | nvidia-cuda-mps-control
        done
    done
done

export TENSORMD_STRATEGY=Scale
model=$(realpath models/SSE_k32_m4_rc6.0_fp32.npz)
for MPS in 8; do
    taskdir="${results_dir}/Maximum/SSE_k32_m4_rc6.0_fp32_MPS${MPS}"
    if [ ! -d "$taskdir" ]; then
        mkdir -v -p "$taskdir"
    fi
    OMP_THREADS=6
    export OMP_NUM_THREADS=$OMP_THREADS
    nvidia-cuda-mps-control -d > /dev/null 2>&1
    mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
        -in in.lammps \
        -var model $model \
        -var nx 36 \
        -var ny 36 \
        -var nz 32 \
        -log "${taskdir}/log.lammps"
    echo quit | nvidia-cuda-mps-control
done
