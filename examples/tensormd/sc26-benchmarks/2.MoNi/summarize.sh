#!/bin/bash

echo "========================================"
echo "MoNi Benchmarks Summary"
echo "========================================"
echo ""

model_arch="m3k32"

for precision in 32; do
    for strategy in "Baseline" "Production" "Speed" "Scale"; do
        for projectin in "reducible" "irreducible"; do
            for MPS in 1; do
                taskdir="results/${model_arch}/MoNi_${strategy}_${projectin}_fp${precision}_MPS${MPS}"
                if [ ! -f "${taskdir}/log.lammps" ]; then
                    continue
                fi
                speed=$(grep "Performance:.*ns/day.*" "${taskdir}/log.lammps" | awk '{print $8}')
                memory=$(grep "Total Memory.*MB" "${taskdir}/log.lammps" | awk '{print $3}')
                natoms=$(grep "Loop time of .* with .* atoms" "${taskdir}/log.lammps" | awk '{print $12}')
                footprint=$(bc <<< "scale=2; $memory * $MPS * 1024 / $natoms")
                printf "Strategy=%-10s Precision=%2d Projection=%-12s MPS=%-8s: %9.4f Matoms/s %8.2f kB/atom\n" \
                    "$strategy"  "$precision" "$projectin" "${MPS}" "$speed" "$footprint"
            done
        done
    done
done

echo "========================================"
echo "Maximum Performance with Scale Strategy:"
echo "========================================"
echo ""

for precision in 32; do
    strategy="Scale"
    projectin="irreducible"
    for MPS in 1 8; do
        taskdir="results/Maximum/MoNi_${projectin}_fp${precision}_MPS${MPS}"
        if [ ! -f "${taskdir}/log.lammps" ]; then
            continue
        fi
        speed=$(grep "Performance:.*ns/day.*" "${taskdir}/log.lammps" | awk '{print $8}')
        memory=$(grep "Total Memory.*MB" "${taskdir}/log.lammps" | awk '{print $3}')
        natoms=$(grep "Loop time of .* with .* atoms" "${taskdir}/log.lammps" | awk '{print $12}')
        footprint=$(bc <<< "scale=2; $memory * $MPS * 1024 / $natoms")
        printf "Strategy=%-10s Precision=%2d Projection=%-12s MPS=%-8s: %9.4f Matoms/s %8.2f kB/atom\n" \
            "$strategy"  "$precision" "$projectin" "${MPS}" "$speed" "$footprint"
    done
done
