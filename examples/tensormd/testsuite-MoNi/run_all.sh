#!/bin/bash
# This script runs all the test cases in the AlCuMg testsuite for TensorMD.
#
# User must provide two environment variables before running this script:
# 1. TENSORMD_BIN: the path to the TensorMD executable
# 2. PYTHON_BIN: the path to the Python executable for running the verification script.
# 
# This script will produce four log files in the current directory:
# 1. diff_cpu_interp0.log: results for CPU device with interpolation delta = 0
# 2. diff_cpu_interp1.log: results for CPU device with interpolation delta = 0.0001
# 3. diff_gpu_interp0.log: results for GPU device with interpolation delta = 0
# 4. diff_gpu_interp1.log: results for GPU device with interpolation delta = 0.0001
#
# dE (eV), dF (eV/ang) and dP (bar) values in the log files should be suficiently 
# small (e.g., < 1e-6) to confirm the correctness of the TensorMD implementation.

testsuite_dir=$(pwd -P)
lammps_root_dir=${testsuite_dir}/../../..

if [ -z $TENSORMD_BIN ]; then
    echo "Error: TENSORMD_BIN environment variable is not set."
    exit 1
else
    if [ ! -x $TENSORMD_BIN ]; then
        echo "Error: TENSORMD_BIN is not an executable file."
        exit 1
    fi
    echo "Using TensorMD binary: $TENSORMD_BIN"
fi

if [ -z $PYTHON_BIN ]; then 
    echo "Error: PYTHON_BIN environment variable is not set."
    exit 1
fi

function cleanup() {
    rm -f c*_pot-MoNi_*/*/log.lammps
    rm -f c*_pot-MoNi_*/*/results.*
}

echo " ****************************************************************"
echo " CPU / Interp delta = 0"
echo " ****************************************************************"
export TENSORMD_DEVICE=cpu
export TENSORMD_INTERP_DELTA=0
cleanup
for case_dir in ./c*_pot-*; do
    cd $case_dir
    for actfn in softplus tanh silu silu32; do
        cd ${actfn}
        $TENSORMD_BIN -in in.lammps -log none -screen none
        echo "Finished case: $case_dir / ${actfn}"
        cd ..
    done
    cd $testsuite_dir
done
echo "All cases finished for CPU / Interp delta = 0"
echo ""
$PYTHON_BIN verify_results.py > diff_cpu_interp0.log

echo " ****************************************************************"
echo " CPU / Interp delta = 0.0001"
echo " ****************************************************************"
export TENSORMD_DEVICE=cpu
export TENSORMD_INTERP_DELTA=0.0001
cleanup
for case_dir in ./c*_pot-*; do
    cd $case_dir
    for actfn in softplus tanh silu silu32; do
        cd ${actfn}
        $TENSORMD_BIN -in in.lammps -log none -screen none
        echo "Finished case: $case_dir / ${actfn}"
        cd ..
    done
    cd $testsuite_dir
done
echo "All cases finished for CPU / Interp delta = 0.0001"
echo ""
$PYTHON_BIN verify_results.py > diff_cpu_interp1.log

echo " ****************************************************************"
echo " GPU / Interp delta = 0"
echo " ****************************************************************"
export TENSORMD_DEVICE=gpu
export TENSORMD_INTERP_DELTA=0
cleanup
for case_dir in ./c*_pot-*; do
    cd $case_dir
    for actfn in softplus tanh silu silu32; do
        cd ${actfn}
        $TENSORMD_BIN -in in.lammps -log none -screen none
        echo "Finished case: $case_dir / ${actfn}"
        cd ..
    done
    cd $testsuite_dir
done
echo "All cases finished for GPU / Interp delta = 0"
echo ""
$PYTHON_BIN verify_results.py > diff_gpu_interp0.log

echo " ****************************************************************"
echo " GPU / Interp delta = 0.0001"
echo " ****************************************************************"
export TENSORMD_DEVICE=gpu
export TENSORMD_INTERP_DELTA=0.0001
cleanup
for case_dir in ./c*_pot-*; do
    cd $case_dir
    for actfn in softplus tanh silu silu32; do
        cd ${actfn}
        $TENSORMD_BIN -in in.lammps -log none -screen none
        echo "Finished case: $case_dir / ${actfn}"
        cd ..
    done
    cd $testsuite_dir
done
echo "All cases finished for GPU / Interp delta = 0.0001"
$PYTHON_BIN verify_results.py > diff_gpu_interp1.log
