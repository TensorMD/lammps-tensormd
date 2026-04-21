#!/bin/bash

for ntypes in 1 2 3 4 5 6; do

    echo "========================="
    echo "Summary of HEA benchmarks with ntypes=${ntypes}"
    echo "========================="

    for taskdir in results/ntypes${ntypes}/*; do
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
        footprint=$(bc <<< "scale=2; $memory * $MPS * 1024 / $natoms")
        printf "ntypes=%-4s k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
            "${ntypes}" "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
    done
done

echo "========================="
echo "Policy Ablation"
echo "========================="
for taskdir in results/PolicyAblation/*/*; do
    if [ ! -f "${taskdir}/log.lammps" ]; then
        continue
    fi
    policy=$(echo "${taskdir}" | grep -o "PolicyAblation/[^/]\+" | cut -d "/" -f 2)
    MPS=$(echo "${taskdir}" | grep -o "MPS[0-9]\+" | cut -d "S" -f 2)
    k_dim=$(echo "${taskdir}" | grep -o "k[0-9]\+" | cut -d "k" -f 2)
    m_dim=$(echo "${taskdir}" | grep -o "m[0-9]\+" | cut -d "m" -f 2)
    rcut=$(echo "${taskdir}" | grep -o "rc[0-9]\+\.[0-9]\+" | cut -d "c" -f 2)
    hybrid=$(echo "${taskdir}" | grep -o "hybrid[01]" | cut -d "d" -f 2)
    speed=$(grep "Performance:.*ns/day.*" "${taskdir}/log.lammps" | awk '{print $8}')
    memory=$(grep "Total Memory.*MB" "${taskdir}/log.lammps" | awk '{print $3}')
    natoms=$(grep "Loop time of .* with .* atoms" "${taskdir}/log.lammps" | awk '{print $12}')
    footprint=$(bc <<< "scale=2; $memory * $MPS * 1024 / $natoms")
    printf "%10s k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
        "${policy}" "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
done
