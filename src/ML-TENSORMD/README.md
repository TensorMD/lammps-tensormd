# Sunway
## Compile command example

```bash
export CC=mpicc
export CXX=mpic++

export CPLUS_INCLUDE_PATH=/usr/sw/yyzlib/lapack-3.8.0/CBLAS/include 
cmake ../cmake \
    -C ../cmake/presets/tensormd.cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_MPI=yes \
    -D BUILD_OMP=no \
    -D BUILD_SW=yes \
    -D BUILD_BASELINE=no \
    -D USE_SW_NNP=yes \
    -D SW_KERNEL=../src/ML-TENSORMD/sw_kernel/lib \
    -D BLAS_LIBRARIES=/usr/sw/yyzlib/xMath.bak/210316/libswblas.a \
    -D CMAKE_FIND_ROOT_PATH=/usr/sw/swgcc/swgcc710-tools-SEA-1079/usr
```

- `-DBUILD_SW=yes` for Sunway implementation.
- set `CPLUS_INCLUDE_PATH` to the path of include directory of CBLAS.
- `-DUSE_SW_NNP={yes,no}` enable optimized NN kernel with better performance, but need to set NN config manually (you can choose predefined config set in `src/ML-TENSORMD/tensormd.h`).
- `-DBUILD_BASELINE={yes, no}` to enable baseline implementation.
- For large scaling test, set `-DLAMMPS_SIZES=bigbig`.

## Run command example

```bash
#run
bsub -I -J TensorMD -q q_sw_expr -b -n 6 -cgsp 64 -share_size 15500 ../lammps/build/lmp -in in.lammps -log log.lammps
```
