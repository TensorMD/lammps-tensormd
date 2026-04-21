#!/bin/bash

#SBATCH --job-name=MoNi
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1

module load cuda-12.4 tbb/latest compiler-rt/2024.2.0 compiler/2024.2.0 mpi/2021.13 mkl/2024.2
export PATH=/share/home/chenxin/software/tensormd/dev/build/release:$PATH

export TENSORMD_INTERP_DELTA=0.01
export TENSORMD_VERBOSE_DEVICE_MAP=false
export TENSORMD_MEMORY_GROWTH_RATE=1.05
export TENSORMD_TIMER=on
export TENSORMD_DEVICE=gpu

results_dir=results
if [ ! -d "$results_dir" ]; then
    mkdir -v "$results_dir"
fi

model_arch="m3k32"

# for precision in "fp32"; do
#     echo "Running benchmarks with precision: $precision"
#     for strategy in "Baseline" "Speed" "Production"; do
#         export TENSORMD_STRATEGY=$strategy
#         echo "  Running with strategy: $strategy"
#         for projectin in "reducible" "irreducible"; do
#             model=$(realpath models/MoNi_${model_arch}_${projectin}_${precision}.npz)
#             if [ ! -f "$model" ]; then
#                 echo "Model file not found: $model"
#                 continue
#             fi
#             echo "    Running with projection: $projectin"
#             for MPS in 1; do
#                 export TENSORMD_USE_HYBRID_KERNEL=yes
#                 taskdir="${results_dir}/${model_arch}/MoNi_${strategy}_${projectin}_${precision}_MPS${MPS}"
#                 if [ ! -d "$taskdir" ]; then
#                     mkdir -v -p "$taskdir"
#                 fi
#                 echo "      Running with MPS: $MPS"
#                 OMP_THREADS=6
#                 nvidia-cuda-mps-control -d > /dev/null 2>&1
#                 mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
#                     -in in.lammps \
#                     -var model $model \
#                     -log "${taskdir}/log.lammps" \
#                     -var nx 40
#                 echo quit | nvidia-cuda-mps-control
#                 sleep 1
#             done
#         done
#     done
# done

export TENSORMD_STRATEGY=Scale
model=$(realpath models/MoNi_m2k32L64N2rc4.5_irreducible_fp32.npz)
for MPS in 8; do
    taskdir="${results_dir}/Maximum2/MoNi_irreducible_fp32_MPS${MPS}"
    if [ ! -d "$taskdir" ]; then
        mkdir -v -p "$taskdir"
    fi
    nvidia-cuda-mps-control -d > /dev/null 2>&1
    OMP_THREADS=6
    export OMP_NUM_THREADS=$OMP_THREADS
    mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
        -in in.lammps \
        -var model $model \
        -log "${taskdir}/log.lammps" \
        -var nx 143
    echo quit | nvidia-cuda-mps-control
done
