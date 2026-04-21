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

results_dir=results
if [ ! -d "$results_dir" ]; then
    mkdir -v "$results_dir"
fi

precision="fp32"

# First pass: beta-Sn, 0GPa, 40x40x40 supercell
export TENSORMD_STRATEGY=Production
for k in 32 64; do
    for m in 4 6; do
        for rcut in 6.0 7.0 8.0; do
            model=$(realpath models/Sn_k${k}_m${m}_rc${rcut}_${precision}.npz)
            if [ ! -f "$model" ]; then
                echo "Model file not found: $model"
                continue
            fi
            for MPS in 1 8; do
                for hybrid in "1" "0"; do
                    export TENSORMD_USE_HYBRID_KERNEL=$hybrid
                    taskdir="${results_dir}/beta/Production/Sn_k${k}_m${m}_rc${rcut}_${precision}_MPS${MPS}_hybrid${hybrid}"
                    if [ ! -d "$taskdir" ]; then
                        mkdir -v -p "$taskdir"
                    fi
                    OMP_THREADS=6
                    nvidia-cuda-mps-control -d > /dev/null 2>&1
                    mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
                        -in in.lammps \
                        -var model $model \
                        -var phase beta \
                        -var nx 40 \
                        -var ny 40 \
                        -var nz 40 \
                        -log "${taskdir}/log.lammps"
                    echo quit | nvidia-cuda-mps-control
                done
            done
        done
    done
done

export TENSORMD_STRATEGY=Speed
for k in 32 64; do
    for m in 4; do
        for rcut in 6.0; do
            model=$(realpath models/Sn_k${k}_m${m}_rc${rcut}_${precision}.npz)
            if [ ! -f "$model" ]; then
                echo "Model file not found: $model"
                continue
            fi
            for MPS in 1; do
                for hybrid in "1"; do
                    export TENSORMD_USE_HYBRID_KERNEL=$hybrid
                    taskdir="${results_dir}/beta/Speed/Sn_k${k}_m${m}_rc${rcut}_${precision}_MPS${MPS}_hybrid${hybrid}"
                    if [ ! -d "$taskdir" ]; then
                        mkdir -v -p "$taskdir"
                    fi
                    OMP_THREADS=6
                    nvidia-cuda-mps-control -d > /dev/null 2>&1
                    mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
                        -in in.lammps \
                        -var model $model \
                        -var phase beta \
                        -var nx 40 \
                        -var ny 40 \
                        -var nz 40 \
                        -log "${taskdir}/log.lammps"
                    echo quit | nvidia-cuda-mps-control
                done
            done
        done
    done
done

for strategy in "Baseline" "Speed" "Scale"; do
    export TENSORMD_STRATEGY=$strategy
    model=$(realpath models/Sn_k32_m4_rc6.0_fp32.npz)
    if [ ! -f "$model" ]; then
        echo "Model file not found: $model"
        continue
    fi
    for MPS in 1; do
        export TENSORMD_USE_HYBRID_KERNEL=yes
        taskdir="${results_dir}/beta/PolicyAblation/${strategy}/Sn_k32_m3_rc6.0_fp32_MPS${MPS}_hybrid1"
        if [ ! -d "$taskdir" ]; then
            mkdir -v -p "$taskdir"
        fi
        OMP_THREADS=6
        # nvidia-cuda-mps-control -d > /dev/null 2>&1
        mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
            -in in.lammps \
            -var model $model \
            -log "${taskdir}/log.lammps" \
            -var phase beta \
            -var nx 80 \
            -var ny 40 \
            -var nz 40
        # echo quit | nvidia-cuda-mps-control
        sleep 1
    done
done

# # Second pass: beta-Sn, 0GPa, 40x40x40 supercell
# export TENSORMD_STRATEGY=Baseline
for k in 32; do
    for m in 4; do
        for rcut in 6.0 7.0; do
            model=$(realpath models/Sn_k${k}_m${m}_rc${rcut}_${precision}.npz)
            if [ ! -f "$model" ]; then
                echo "Model file not found: $model"
                continue
            fi
            for MPS in 1 8; do
                export TENSORMD_USE_HYBRID_KERNEL=true
                taskdir="${results_dir}/beta/Baseline/Sn_k${k}_m${m}_rc${rcut}_${precision}_MPS${MPS}_hybrid1"
                if [ ! -d "$taskdir" ]; then
                    mkdir -v -p "$taskdir"
                fi
                OMP_THREADS=6
                nvidia-cuda-mps-control -d > /dev/null 2>&1
                mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
                    -in in.lammps \
                    -var model $model \
                    -var phase beta \
                    -var nx 60 \
                    -var ny 60 \
                    -var nz 60 \
                    -log "${taskdir}/log.lammps"
                echo quit | nvidia-cuda-mps-control
            done
        done
    done
done

# # Third pass: beta-Sn, 0GPa, 40x40x40 supercell, k=32, m=4/6, rcut=6.0/7.0/8.0, MPS=1/8, hybrid=1/0
# # Strategy: Scale
export TENSORMD_STRATEGY=Scale
for m in 4 6; do
    for rcut in 6.0; do
        model=$(realpath models/Sn_k32_m${m}_rc${rcut}_${precision}.npz)
        if [ ! -f "$model" ]; then
            echo "Model file not found: $model"
            continue
        fi
        for MPS in 1 8; do
            export TENSORMD_USE_HYBRID_KERNEL=true
            taskdir="${results_dir}/beta/Scale/Sn_k32_m${m}_rc${rcut}_${precision}_MPS${MPS}_hybrid1"
            if [ ! -d "$taskdir" ]; then
                mkdir -v -p "$taskdir"
            fi
            OMP_THREADS=6
            nvidia-cuda-mps-control -d > /dev/null 2>&1
            mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
                -in in.lammps \
                -var model $model \
                -var phase beta \
                -var nx 40 \
                -var ny 40 \
                -var nz 40 \
                -log "${taskdir}/log.lammps"
            echo quit | nvidia-cuda-mps-control
        done
    done
done

# # Fourth pass: delta-Sn, alat=3.4/3.3/3.2/3.1/3.0, 40x40x40 supercell, k=32, m=4/6, rcut=6.0/7.0/8.0, MPS=1/8, hybrid=1
# Strategies: Scale, Production
for strategy in "Scale" "Production"; do
    export TENSORMD_STRATEGY=$strategy
    for alat in 3.4 3.3 3.2 3.1 3.0; do
        for m in 4 6; do
            for rcut in 6.0 7.0 8.0; do
                model=$(realpath models/Sn_k32_m${m}_rc${rcut}_${precision}.npz)
                if [ ! -f "$model" ]; then
                    echo "Model file not found: $model"
                    continue
                fi
                for MPS in 1 8; do
                    export TENSORMD_USE_HYBRID_KERNEL=true
                    taskdir="${results_dir}/delta/${strategy}/Sn_alat${alat}_k32_m${m}_rc${rcut}_${precision}_MPS${MPS}_hybrid1"
                    if [ ! -d "$taskdir" ]; then
                        mkdir -v -p "$taskdir"
                    fi
                    OMP_THREADS=6
                    nvidia-cuda-mps-control -d > /dev/null 2>&1
                    mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
                        -in in.lammps \
                        -var model $model \
                        -var phase delta \
                        -var alat $alat \
                        -var nx 40 \
                        -var ny 40 \
                        -var nz 40 \
                        -log "${taskdir}/log.lammps"
                    echo quit | nvidia-cuda-mps-control
                done
            done
        done
    done
done

# # Fifth pass: beta-Sn, 0GPa, maximum scale, k=32, m=4, rcut=6.0, MPS=1, Strategy: Scale
export TENSORMD_STRATEGY=Scale
model=$(realpath models/Sn_k32_m4_rc6.0_fp32.npz)
if [ ! -f "$model" ]; then
    echo "Model file not found: $model"
else
    for MPS in 8; do
        taskdir="${results_dir}/beta/Maximum/Sn_k32_m4_rc6.0_fp32_MPS${MPS}"
        if [ ! -d "$taskdir" ]; then
            mkdir -v -p "$taskdir"
        fi
        OMP_THREADS=6
        export OMP_NUM_THREADS=$OMP_THREADS
        nvidia-cuda-mps-control -d > /dev/null 2>&1
        mpiexec -np $MPS lmp_tensormd -pk omp $OMP_THREADS neigh yes \
            -in in.lammps \
            -var model $model \
            -var phase beta \
            -var nx 190 \
            -var ny 190 \
            -var nz 190 \
            -log "${taskdir}/log.lammps"
        echo quit | nvidia-cuda-mps-control
    done
fi
