#!/bin/bash
cd $SCRATCH/OpenThoughts-Agent
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor/src:$SCRATCH/OpenThoughts-Agent
set -a; source $SCRATCH/keys.env; set +a
export EVAL_TACC_N_CONCURRENT_CEILING=12     # a1 is 8B; the default 8 is a 32B-derived ceiling
exec python -m hpc.launch --job_type eval_listener \
  --cluster-config tacc --preset tb2 \
  --datasets $SCRATCH/data/datasets/terminal_bench_2 \
  --require-priority-list --priority-file eval/lists/mrinal_flawed_summ_a1_tb2.txt \
  --baseline-model-configs eval/clusters/mrinal_tacc_a1_8b.yaml \
  --force-reeval --n-concurrent 12 --max-jobs-submitted 4 \
  --check-hours 0.5 --slurm-time 23:59:00 --verbose
