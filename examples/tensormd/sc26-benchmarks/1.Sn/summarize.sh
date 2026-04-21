#!/bin/bash

echo "========================="
echo "Summary of Sn benchmarks"
echo "========================="
echo ""
echo "> Beta-Sn, ~0GPa, 40x40x40 supercell, Production strategy"
echo ""
for taskdir in results/beta/Production/*; do
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
    printf "k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
        "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
done

echo ""
echo "> Beta-Sn, ~0GPa, 40x40x40 supercell, Speed strategy"
echo ""
for taskdir in results/beta/Speed/*; do
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
    printf "k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
        "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
done

echo ""
echo "> Beta-Sn, ~0GPa, 40x40x40 supercell, Ablation"
echo ""
for taskdir in results/beta/PolicyAblation/*/*; do
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
    printf "k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
        "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
done

echo ""
echo "> Beta-Sn, ~0GPa, 40x40x40 supercell, Baseline strategy"
echo ""
for taskdir in results/beta/Baseline/*; do
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
    printf "k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
        "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
done

echo ""
echo "> Beta-Sn, ~0GPa, 40x40x40 supercell, Scale strategy"
echo ""

for taskdir in results/beta/Scale/*; do
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
    printf "k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
        "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
done

echo ""
echo "> Delta-Sn, 40x40x40 supercell, Scale strategy"
echo ""

for taskdir in results/delta/Scale/*; do
    if [ ! -f "${taskdir}/log.lammps" ]; then
        continue
    fi
    alat=$(echo "${taskdir}" | grep -o "alat[0-9]\+\.[0-9]\+" | cut -d "t" -f 2)
    if [ "${alat}" == "3.4" ]; then
        press=35
    elif [ "${alat}" == "3.3" ]; then
        press=55
    elif [ "${alat}" == "3.2" ]; then
        press=85
    elif [ "${alat}" == "3.1" ]; then
        press=125
    else
        press=185
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
    printf "alat=%-4s press=%-4s k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
        "${alat}" "${press}" "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
done

echo ""
echo "> Delta-Sn, 40x40x40 supercell, Production strategy"
echo ""

for taskdir in results/delta/Production/*; do
    if [ ! -f "${taskdir}/log.lammps" ]; then
        continue
    fi
    alat=$(echo "${taskdir}" | grep -o "alat[0-9]\+\.[0-9]\+" | cut -d "t" -f 2)
    if [ "${alat}" == "3.4" ]; then
        press=35
    elif [ "${alat}" == "3.3" ]; then
        press=55
    elif [ "${alat}" == "3.2" ]; then
        press=85
    elif [ "${alat}" == "3.1" ]; then
        press=125
    else
        press=185
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
    printf "alat=%-4s press=%-4s k=%-4s m=%-4s rcut=%-4s MPS=%-8s hybrid=%-4s: %9.4f Matoms/s %8.2f kB/atom\n" \
        "${alat}" "${press}" "${k_dim}" "${m_dim}" "${rcut}" "${MPS}" "${hybrid}" "$speed" "$footprint"
done
