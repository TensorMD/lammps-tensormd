#!/bin/sh
set -x
#cp ../main ./
now_time=$(date "+%Y%m%d%H%M%S")
for exe in kernel1_main kernel2_main kernel3_main kernel4_main 
	do 
    bsub -I -shared -b -o out.log -q q_9a_8000 -n 1 -cgsp 64 -node 27 -share_size 8192 ./$exe -a500 -b1 -c200  -k64 
	done

	bsub -I -b -q q_9a_8000 -n 1 -cgsp 64 -share_size 12000 -host_stack 2048 -cache_size 0 ./nnp_main 2>&1 | tee run.log
