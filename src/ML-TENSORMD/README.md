## Requirements
- G++ / ICPX
- MPI
- CMake
- CUDA Toolkit / ROCm

## Compile command example

```bash
cd lammps
mkdir build
cd build

export CC=mpicc
export CXX=mpicxx
cmake ../cmake \
    -C ../cmake/presets/tensormd.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_MPI=yes \
    -DBUILD_OMP=yes \
    -DBUILD_CUDA=yes \
    -DBUILD_HIP=no \
    -DBUILD_BASELINE=no \
    -DENABLE_CUDA_WMMA=yes \
    -DUSE_MKL_BATCH=yes \
    -DLINK_SHARED=no
make -j8
```

- `-DBUILD_CUDA=yes; -DBUILD_HIP=no` for CUDA implementation, `-DBUILD_CUDA=no; -DBUILD_HIP=yes` for HIP implementation.
- `-DLINK_SHARED={yes, no}`: `yes` for dynamic linking, `no` for static linking, will not affect performance.
- If `-DBUILD_CUDA=yes`, for NVIDIA GPUs supporting FP64 TensorCore (Ampere architecture or later), set `ENABLE_CUDA_WMMA=yes` for better performance.
- `-DBUILD_BASELINE={yes, no}` to enable baseline implementation, other performance options will not afect baseline implementation.
- For large scaling test, set `-DLAMMPS_SIZES=bigbig`.
- It is recommended to keep the other option settings in the example unchanged.

## Run command example
```bash
# run
export OMP_NUM_THREADS=40
mpirun -np 1 /home/oyyc/lammps/build/lmp -in in.lammps -log log.lammps
# nsys profiling
nsys profile --stats=true mpirun -np 1 /home/oyyc/lammps/build/lmp -in in.lammps -log log.lammps
# ncu profiling
ncu --target-processes all -o nculog mpirun -np 1 /home/oyyc/lammps/build/lmp -in in.lammps -log log.lammps
```

# ORISE

## Compile command example

```bash
cd lammps
mkdir build
cd build

module use /public/software/modules
module load compiler/rocm/dtk/22.10.1
module load compiler/cmake/3.24.1
module load compiler/llvm-openmp/openmp

export CC=mpicc
export CXX=mpicxx
cmake ../cmake \
    -C ../cmake/presets/tensormd.cmake \
    -D BLAS_INCLUDE_DIRS="/public/software/mathlib/openblas/0.3.7/include" \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=yes \
    -D BUILD_OMP=yes \
    -D BUILD_HIP=yes \
    -D BUILD_CUDA=no \
    -D BUILD_BASELINE=no \
    -D USE_MKL_BATCH=yes \
    -D LINK_SHARED=no
make -j8
```

- `-DBUILD_CUDA=yes; -DBUILD_HIP=no` for CUDA implementation, `-DBUILD_CUDA=no; -DBUILD_HIP=yes` for HIP implementation.
- `-DLINK_SHARED={yes, no}`: `yes` for dynamic linking, `no` for static linking.
- `-DBUILD_BASELINE={yes, no}` to enable baseline implementation.
- For large scaling test, set `-DLAMMPS_SIZES=bigbig`.


## Slurm example


```bash
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4

export OMP_NUM_THREADS=6
mpirun -np 1 /home/oyyc/lammps/build/lmp -in in.lammps -log log.lammps
```
