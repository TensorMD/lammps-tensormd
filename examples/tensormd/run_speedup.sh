#!/bin/bash

export OMP_NUM_THREADS=40

output_file="speedup.csv"

echo " ,Flexible neighbor-list padding (Section 5.4),GEMM efficiency enhancement (Section 5.3),Forward-backward strength reduction (Section 5.1)" > "$output_file"


if [ -z "$TENSORMD_BASELINE_PATH" ]; then
    echo "TENSORMD_BASELINE_PATH is empty."
    exit 1
else
    echo "TENSORMD_BASELINE_PATH is: $TENSORMD_BASELINE_PATH"
fi

if [ -z "$TENSORMD_PATH" ]; then
    echo "TENSORMD_PATH is empty."
    exit 1
else
    echo "TENSORMD_PATH is: $TENSORMD_PATH"
fi

echo -n "W" >> "$output_file"

cd ./W/tensormd/speedup
echo "Enter $PWD"

logfile="log1.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_BASELINE_PATH -in in_default.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
baseline=$(echo "$pair_line" | awk '{print $5}')

logfile="log2.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_BASELINE_PATH -in in.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
step1=$(echo "$pair_line" | awk '{print $5}')
pair_line=$(grep "^   fnn->interp" "$logfile" 2>/dev/null)
ration_old=$(echo "$pair_line" | awk '{print $4}')
result=$(echo "scale=4; $baseline / $step1" | bc)
echo -n ",$result" >> "../../../$output_file"

logfile="log3.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_PATH -in in.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
step2=$(echo "$pair_line" | awk '{print $5}')
pair_line=$(grep "^   fnn->interp" "$logfile" 2>/dev/null)
ration=$(echo "$pair_line" | awk '{print $4}')
result=$(echo "scale=4; $baseline / ($step2 - $ration + $ration_old)" | bc)
echo -n ",$result" >> "../../../$output_file"

result=$(echo "scale=4; $baseline / $step2" | bc)
echo -n ",$result" >> "../../../$output_file"

echo "" >> "../../../$output_file"
cd ../../..

echo -n "MoNi" >> "$output_file"

cd ./binary/medium
echo "Enter $PWD"

logfile="log1.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_BASELINE_PATH -in in_default.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
baseline=$(echo "$pair_line" | awk '{print $5}')

logfile="log2.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_BASELINE_PATH -in in.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
step1=$(echo "$pair_line" | awk '{print $5}')
pair_line=$(grep "^   fnn->interp" "$logfile" 2>/dev/null)
ration_old=$(echo "$pair_line" | awk '{print $4}')
result=$(echo "scale=4; $baseline / $step1" | bc)
echo -n ",$result" >> "../../$output_file"

logfile="log3.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_PATH -in in.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
step2=$(echo "$pair_line" | awk '{print $5}')
pair_line=$(grep "^   fnn->interp" "$logfile" 2>/dev/null)
ration=$(echo "$pair_line" | awk '{print $4}')
result=$(echo "scale=4; $baseline / ($step2 - $ration + $ration_old)" | bc)
echo -n ",$result" >> "../../$output_file"

result=$(echo "scale=4; $baseline / $step2" | bc)
echo -n ",$result" >> "../../$output_file"

echo "" >> "../../$output_file"
cd ../..

echo -n "AlMgSi" >> "$output_file"

cd ./trinary
echo "Enter $PWD"

logfile="log1.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_BASELINE_PATH -in in_default.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
baseline=$(echo "$pair_line" | awk '{print $5}')

logfile="log2.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_BASELINE_PATH -in in.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
step1=$(echo "$pair_line" | awk '{print $5}')
pair_line=$(grep "^   fnn->interp" "$logfile" 2>/dev/null)
ration_old=$(echo "$pair_line" | awk '{print $4}')
result=$(echo "scale=4; $baseline / $step1" | bc)
echo -n ",$result" >> "../$output_file"

logfile="log3.lammps"
echo "Generating file: $logfile"
mpirun -np 1 $TENSORMD_PATH -in in.lammps -log $logfile -screen none
pair_line=$(grep "^Pair" "$logfile" 2>/dev/null)
step2=$(echo "$pair_line" | awk '{print $5}')
pair_line=$(grep "^   fnn->interp" "$logfile" 2>/dev/null)
ration=$(echo "$pair_line" | awk '{print $4}')
result=$(echo "scale=4; $baseline / ($step2 - $ration + $ration_old)" | bc)
echo -n ",$result" >> "../$output_file"

result=$(echo "scale=4; $baseline / $step2" | bc)
echo -n ",$result" >> "../$output_file"

echo "" >> "../$output_file"
cd ..

echo "Result CSV file: $output_file"