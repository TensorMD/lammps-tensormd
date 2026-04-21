#!/bin/bash

function extract_metrics() {
    model=$1
    log_file=$2
    throughput=$(grep "Performance:.*ns/day.*" "${log_file}" | awk '{print $8}')
    natoms=$(grep "Loop time of .* with .* atoms" "${log_file}" | awk '{print $12}')
    pair_ratio=$(grep "Pair.*|.*" "${log_file}" | awk '{print $11}')
    comm_ratio=$(grep "Comm.*|.*" "${log_file}" | awk '{print $11}')
    modify_ratio=$(grep "Modify.*|.*" "${log_file}" | awk '{print $11}')
    printf "%9s  %8d    %8.2f   %5.2f%%   %5.2f%%   %5.2f%%\n" \
        "${model}" "$natoms" "$throughput" "$pair_ratio" "$comm_ratio" "$modify_ratio"
}

printf "%9s  %8s   %8s   %5s   %5s   %5s\n" "Model" "NAtoms" "Throughput" "Pair%" "Comm%" "Modify%"

for log_file in results/natoms_vs_speed/log.*.lammps; do
    model="V2"
    extract_metrics "${model}" "${log_file}"
done | sort -k2 -n

for log_file in V1/ambient/log.*.lammps; do
    model="V1"
    extract_metrics "${model}" "${log_file}"
done | sort -k2 -n
