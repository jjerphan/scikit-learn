#!/usr/bin/env bash
#
# Launch a ASV benchmark on groups of CPU
# for scalability inspection
#
# Results are saved in given folders and files.
#
for i in 128 64 32 16 8 4 2 1;
do
  last_core=$(($i-1))
    taskset -c 0-$last_core \
    asv continuous -b PairwiseDistancesArgKmin \
    -e main feat/pdr-32bit | tee pairwise_distances_argkmin_asv_${i}_cores.txt
    cp -R results results_${i}_cores
done



