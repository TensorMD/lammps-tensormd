#!/bin/bash


echo "========================="
echo "Summary of AlCuMg benchmarks"
echo "========================="

for taskdir in results/*; do
    if [ ! -f "${taskdir}/log.lammps" ]; then
        continue
    fi
    MPS=$(echo "${taskdir}" | grep -o "MPS[0-9]\+" | cut -d "S" -f 2)
    k_dim=$(echo "${taskdir}" | grep -o "k[0-9]\+" | cut -d "k" -f 2)
    m_dim=$(echo "${taskdir}" | grep -o "m[0-9]\+" | cut -d "m" -f 2)
    rcut=$(echo "${taskdir}" | grep -o "rc[0-9]\+\.[0-9]\+" | cut -d "c" -f 2)
    hybrid=$(echo "${taskdir}" | grep -o "hybrid[01]" | cut -d "d" -f 2)
    speed=$(grep "Performance:.*ns/day.*" "${taskdir}/log.lammps" | awk '{print $8}')
    memory=$(grep "Total Memory.*MB" "${taskdir}/log.lammps" | awk '{print $3}')
    natoms=$(grep "Loop time of .* with .* atoms" "${taskdir}/log.lammps" | awk '{print $12}')
    total_memory=$(bc <<< "scale=2; $natoms / 1000.0 / ($memory * $MPS / 1024)")
    printf "k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f Katoms/GB\n" \
        "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$total_memory"
done
