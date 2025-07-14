#!/bin/bash

export OMP_NUM_THREADS=40

output_file="compare.csv"

echo " ,20340 atoms,46080 atoms,69120 atoms,92160 atoms,115200 atoms" > "$output_file"

if [ -z "$DP_PATH" ]; then
    echo "DP_PATH is empty."
    exit 1
else
    echo "DP_PATH is: $DP_PATH"
    cd ./W/dp/compare
    echo -n "DP" >> "../../../$output_file"
    for i in {1..5}; do
      logfile="log${i}.lammps"
      echo "Generating file: $logfile"
      mpirun -np 1 $DP_PATH -in in${i}.lammps -log $logfile -screen none
      pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
      fifth_field=$(echo "$pair_line" | awk '{print $5}')
      echo -n ",$fifth_field" >> "../../../$output_file"
    done
    echo "" >> "../../../$output_file"
    cd ../../..
fi

if [ -z "$TENSORMD_PATH" ]; then
    echo "TENSORMD_PATH is empty."
    exit 1
else
    echo "TENSORMD_PATH is: $TENSORMD_PATH"
    cd ./W/tensormd/compare
    echo -n "TensorMD" >> "../../../$output_file"
    for i in {1..5}; do
      logfile="log${i}.lammps"
      echo "Generating file: $logfile"
      mpirun -np 1 $TENSORMD_PATH -in in${i}.lammps -log $logfile -screen none
      pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
      fifth_field=$(echo "$pair_line" | awk '{print $5}')
      echo -n ",$fifth_field" >> "../../../$output_file"
    done
    echo "" >> "../../../$output_file"
fi

echo "Result CSV file: $output_file"