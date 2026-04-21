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

results_dir=results
if [ ! -d "$results_dir" ]; then
    mkdir -v "$results_dir"
fi

models=(
    models/AlCuMg_k32_m4_rc6.0_L64_fp32.npz
    models/AlCuMg_k64_m6_rc6.0_L256_fp32.npz
)

for model in "${models[@]}"; do
    k=$(echo "$model" | grep -o "k[0-9]\+" | cut -d "k" -f 2)
    m=$(echo "$model" | grep -o "m[0-9]\+" | cut -d "m" -f 2)
    model_path=$(realpath "$model")
    for MPS in 1 8; do
        if [ "${k}" == "32" ]; then
            export TENSORMD_USE_HYBRID_KERNEL=1
            taskdir="${results_dir}/AlCuMg_k${k}_m${m}_MPS${MPS}_hybrid1"
        else
            export TENSORMD_USE_HYBRID_KERNEL=0
            taskdir="${results_dir}/AlCuMg_k${k}_m${m}_MPS${MPS}_hybrid0"
        fi
        if [ ! -d "$taskdir" ]; then
            mkdir -v -p "$taskdir"
        fi
        OMP_THREADS=6
        nvidia-cuda-mps-control -d > /dev/null 2>&1
        mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
            -in in.lammps \
            -var model $model_path \
            -var nx 20 \
            -var ny 20 \
            -var nz 20 \
            -log "${taskdir}/log.lammps"
        echo quit | nvidia-cuda-mps-control
    done
done

model=$(realpath models/AlCuMg_k32_m4_rc6.0_L64_fp32.npz)
export TENSORMD_STRATEGY=Scale
for MPS in 8; do
    taskdir="${results_dir}/Maximum1/AlCuMg_k32_m4_rc6.0_L64_fp32_MPS${MPS}"
    if [ ! -d "$taskdir" ]; then
        mkdir -v -p "$taskdir"
    fi
    export OMP_NUM_THREADS=6
    nvidia-cuda-mps-control -d > /dev/null 2>&1
    mpiexec -np $MPS lmp_tensormd -pk omp $OMP_NUM_THREADS neigh yes \
        -in in.lammps \
        -var model $model \
        -var nx 80 \
        -var ny 80 \
        -var nz 75 \
        -log "${taskdir}/log.lammps"
    echo quit | nvidia-cuda-mps-control
done
