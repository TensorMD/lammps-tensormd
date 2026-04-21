#!/bin/bash
# This script runs all the test cases in the AlCuMg testsuite for TensorMD.

# ==========================================
# Default Configurations
# ==========================================
TENSORMD_BIN="${TENSORMD_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICES=("cpu" "gpu")
INTERP_DELTAS=("0" "0.0001")
LAMMPS_ROOT="${LAMMPS_ROOT:-}"
STRATEGY="${TENSORMD_STRATEGY:-}"
MIXED_PRECISION="${TENSORMD_USE_FP32_ATLAS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CASE_IDS="all"

function print_help() {
    cat << EOF
Usage: $0 [OPTIONS]

This script runs all the test cases in the AlCuMg testsuite for TensorMD.

Options:
  -t,  --tensormd-bin PATH       Path to the TensorMD executable (default: \$TENSORMD_BIN)
  -p,  --python-bin PATH         Path to the Python executable for verification (default: \$PYTHON_BIN or python3)
  -d,  --devices LIST            Comma-separated list of devices to test (default: cpu,gpu)
  -i,  --interp-deltas LIST      Comma-separated list of interpolation deltas to test (default: 0,0.0001)
  -l,  --lammps-root PATH        Path to LAMMPS root directory (default: ../../..)
  -a,  --extra-args ARGS         Extra arguments to pass to the TensorMD binary
  -s,  --strategy STRATEGY       TensorMD strategy to use (overrides TENSORMD_STRATEGY)
  -m,  --mixed-precision MODE    Mixed precision mode (overrides TENSORMD_USE_FP32_ATLAS)
  -c,  --case-ids LIST           Comma-separated list of case IDs to run (default: 1-11)
  -h,  --help                    Show this help message and exit

Environment Variables:
  TENSORMD_BIN                   Fallback path to the TensorMD executable.
  PYTHON_BIN                     Fallback path to the Python executable.
  LAMMPS_ROOT                    Fallback path to the LAMMPS root.

Output files:
  diff_<device>_interp_<delta>.log: dE, dF, and dP values verification logs.

Example:
  $0 -t /path/to/tensormd -p python3 -d cpu,gpu -i 0,0.0001,0.001 -a "-echo screen"
EOF
}

# ==========================================
# CLI Arguments Parsing
# ==========================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tensormd-bin)    TENSORMD_BIN="$2"; shift 2 ;;
        -p|--python-bin)      PYTHON_BIN="$2"; shift 2 ;;
        -d|--devices)         IFS=',' read -r -a DEVICES <<< "$2"; shift 2 ;;
        -c|--case-ids)        IFS=',' read -r -a CASE_IDS <<< "$2"; shift 2 ;;
        -i|--interp-deltas)   IFS=',' read -r -a INTERP_DELTAS <<< "$2"; shift 2 ;;
        -l|--lammps-root)     LAMMPS_ROOT="$2"; shift 2 ;;
        -s|--strategy)        STRATEGY="$2"; shift 2 ;;
        -m|--mixed-precision) MIXED_PRECISION="$2"; shift 2 ;;
        -a|--extra-args)      EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)            print_help; exit 0 ;;
        *)                    echo "Error: Unknown option $1"; print_help; exit 1 ;;
    esac
done

# ==========================================
# Basic Validations & Setup
# ==========================================
if [ -z "$TENSORMD_BIN" ]; then
    echo "Error: TENSORMD_BIN is not set. Use -t or set the TENSORMD_BIN environment variable."
    exit 1
fi

if [ ! -x "$TENSORMD_BIN" ]; then
    echo "Error: TENSORMD_BIN ($TENSORMD_BIN) is not an executable file."
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then 
    echo "Error: Python executable ($PYTHON_BIN) not found."
    exit 1
fi

# Set dynamic directories
testsuite_dir=$(pwd -P)
if [ -z "$LAMMPS_ROOT" ]; then
    LAMMPS_ROOT="${testsuite_dir}/../../.."
fi

echo "================================================================"
echo " Testing Configuration"
echo "================================================================"
echo " TensorMD Binary : $TENSORMD_BIN"
echo " Python Binary   : $(command -v "$PYTHON_BIN")"
echo " LAMMPS Root     : $LAMMPS_ROOT"
echo " Devices         : ${DEVICES[*]}"
echo " Strategy        : ${STRATEGY}"
echo " Mixed Precision : ${MIXED_PRECISION}"
echo " Interp Deltas   : ${INTERP_DELTAS[*]}"
echo " Case IDs        : ${CASE_IDS[*]}"
if [ -n "$EXTRA_ARGS" ]; then
    echo " Extra Args      : $EXTRA_ARGS"
fi
echo "================================================================"
echo ""

function cleanup() {
    rm -f c*_pot-AlCuMg_*/log.lammps
    rm -f c*_pot-AlCuMg_*/results.*
}

# Remove old diff log files to avoid confusion
rm -f diff_*.log

# ==========================================
# Main Execution Loop
# ==========================================
for dev in "${DEVICES[@]}"; do
    dev="${dev// /}" # Strip spaces defensively
    for delta in "${INTERP_DELTAS[@]}"; do
        delta="${delta// /}" # Strip spaces defensively
        
        # Format delta value safely for filenames (e.g., 0.0001 -> 0_0001)
        safe_delta=$(echo "$delta" | tr '.' '_')
        log_file="diff_${dev}_interp_${safe_delta}.log"
        
        echo " ****************************************************************"
        echo " Device = $dev | Interp delta = $delta"
        echo " ****************************************************************"
        
        export TENSORMD_DEVICE="$dev"
        export TENSORMD_INTERP_DELTA="$delta"
        export TENSORMD_STRATEGY="${STRATEGY:-$TENSORMD_STRATEGY}"
        export TENSORMD_USE_FP32_ATLAS="${MIXED_PRECISION:-$TENSORMD_USE_FP32_ATLAS}"
        
        cleanup
        
        for case_dir in ./c*_pot-*; do
            case_id=$(basename "$case_dir" | cut -d'_' -f1 | tr -d 'c')
            if [ "$CASE_IDS" != "all" ] && [[ ! " ${CASE_IDS[*]} " =~ " ${case_id} " ]]; then
                continue
            fi
            # Skip if no directories match the glob
            [ -d "$case_dir" ] || continue
            
            cd "$case_dir" || exit 1
            
            # $EXTRA_ARGS purposefully left unquoted so it expands functionally
            "$TENSORMD_BIN" -in in.1.lammps -log none -screen none -var LAMMPS_ROOT "${LAMMPS_ROOT}" $EXTRA_ARGS
            "$TENSORMD_BIN" -in in.2.lammps -log none -screen none -var LAMMPS_ROOT "${LAMMPS_ROOT}" $EXTRA_ARGS
            
            echo "Finished case: $case_dir"
            cd "$testsuite_dir" || exit 1
        done
        
        echo "All cases finished for Device = $dev | Interp delta = $delta"
        echo "Verifying results -> $log_file"
        "$PYTHON_BIN" verify_results.py > "$log_file"
        cat $log_file
        echo ""        
    done
done

echo "Testing completed successfully!"
