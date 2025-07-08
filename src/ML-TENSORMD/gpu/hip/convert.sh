#!/bin/bash
path=$(pwd)
files=$(ls $path)
for filename in $files
do
   if [ "${filename##*.}"x = "cuh"x ];then
    mv $filename ${filename%.*}.temp
    hipify-perl ${filename%.*}.temp > ${filename%.*}.cuh
    echo "${filename%.*}.cuh => hip file, old file will be del" 
    rm ${filename%.*}.temp -f
   elif [ "${filename##*.}"x = "cu"x ];then
    hipify-perl $filename > ${filename%.*}.hip
    echo "${filename%.*}.hip => hip file, old file will be del" 
    rm ${filename%.*}.cu -f
   fi
done
echo "current dir covert finish!"