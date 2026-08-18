#!/bin/bash
cd $SCRATCH/OpenThoughts-Agent
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
set -a; source $SCRATCH/keys.env; set +a
while true; do
  python eval/refresh_flawed_summ_list.py terminal_bench_2 mrinal_flawed_summ_a1_tb2.txt 2>&1
  python eval/refresh_flawed_summ_list.py dev_set_v2      mrinal_flawed_summ_a1_devset.txt 2>&1
  sleep 1800
done
