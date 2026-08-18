#!/bin/bash
# POLICY front door: ONE benchmark (tb2), ONE listener, --once per sweep, re-fired on a loop.
# --once (not a long-lived listener) is what POLICY prescribes; looping it also self-heals when
# the login node reaps tmux, which silently stalled the campaign once already.
# TACC: OMIT --baseline-model-configs — the gh200 registry is default-on and gives 32768 +
# timeout_multiplier 2.0, i.e. the ORIGINAL eval conditions, so the harbor fix is the onlyvariable.
cd $SCRATCH/OpenThoughts-Agent
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor/src:$SCRATCH/OpenThoughts-Agent
set -a; source $SCRATCH/keys.env; set +a
export EVAL_UPLOAD_HF_ORG=laion   # we can only write laion/, not DCAgent2/
while true; do
  echo "===== sweep $(date '+%F %T') ====="
  bash $SCRATCH/spike/detect_stalls.sh kill
  bash $SCRATCH/spike/health_check.sh
  bash $SCRATCH/spike/prune_partial_trials.sh
  python eval/refresh_flawed_summ_list.py terminal_bench_2 mrinal_flawed_summ_a1_tb2.txt 2>&1
  bash $SCRATCH/spike/drop_inflight.sh $SCRATCH/OpenThoughts-Agent/eval/lists/mrinal_flawed_summ_a1_tb2.txt
  # Top up to CAP, do not add a fixed batch each sweep. --once + a fixed --max-jobs-submitted
  # grows the queue without bound (observed: 32 jobs / 20 running, over the 20 QOS cap and the
  # <=16 standing rule from .agents/ops/tacc/ops.md).
  CAP=16
  INQ=$(squeue -u "$USER" -h | wc -l)
  SLOTS=$(( CAP - INQ ))
  echo "[capacity] in-queue=$INQ cap=$CAP -> submitting up to $SLOTS"
  if [ "$SLOTS" -le 0 ]; then echo "[capacity] at cap, skipping this sweep"; sleep 1800; continue; fi
  python -m hpc.launch --job_type eval_listener \
    --cluster-config tacc --preset tb2 \
    --datasets $SCRATCH/data/datasets/terminal_bench_2 \
    --require-priority-list --priority-file eval/lists/mrinal_flawed_summ_a1_tb2.txt \
    --config-yaml dcagent_eval_config_no_override.yaml \
    --force-reeval --once --verbose --slurm-time 47:59:00 --max-jobs-submitted "$SLOTS" 2>&1 | grep -iE "Submitted batch|-> Submitted|No eligible|error"
  sleep 1800
done
