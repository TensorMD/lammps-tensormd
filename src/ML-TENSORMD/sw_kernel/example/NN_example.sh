#!/bin/sh
cd /home/export/online1/mdt00/systest/swyyz/hex/2022_Mar/test_debug/sw_zgemm
now_time=$(date "+%Y%m%d%H%M%S")
#mkdir TN_test_$(now_time)_log
echo $now_time
for m in 64 128 256 512 1024 2048 4096
#for m in 64 128 256 512 
	do
    for n in 64 128 256 512 1024 2048 4096
    #for n in 64 128 256 512 
      do
        for k in 64 128 256 512 1024 2048 4096
        #for k in 64 128 256 512 
		      do
#						if  [[ $m -eq $k ]]; then
						   ./NN_run.sh $m $n $k $now_time;
#						else
#						   echo "$m $n $k is can not run!"
#						fi
          done
      done
  done


