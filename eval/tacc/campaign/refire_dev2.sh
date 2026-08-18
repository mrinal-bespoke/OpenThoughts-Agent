#!/bin/bash
# POLICY front door: ONE benchmark (dev_set_v2), ONE listener, --once per sweep, re-fired on a loop.
# tb2 is DRAINED (priority list returned 0) -> campaign moved to dev_set_v2 on 2026-08-10.
# STOCK harbor only. The collect-hook patch is unmerged and must NOT enter the campaign.
cd $SCRATCH/OpenThoughts-Agent
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor/src:$SCRATCH/OpenThoughts-Agent
set -a; source $SCRATCH/keys.env; set +a
export EVAL_UPLOAD_HF_ORG=laion   # we can only write laion/, not DCAgent2/
LIST=eval/lists/mrinal_flawed_summ_a1_dev2.txt
while true; do
  echo "===== sweep $(date '+%F %T') ====="
  bash $SCRATCH/spike/detect_stalls.sh kill
  bash $SCRATCH/spike/health_check.sh
  bash $SCRATCH/spike/prune_partial_trials.sh
  python eval/refresh_flawed_summ_list.py dev_set_v2 mrinal_flawed_summ_a1_dev2.txt 2>&1
  bash $SCRATCH/spike/drop_inflight.sh $SCRATCH/OpenThoughts-Agent/$LIST
  CAP=16
  INQ=$(squeue -u "$USER" -h | wc -l)
  SLOTS=$(( CAP - INQ ))
  echo "[capacity] in-queue=$INQ cap=$CAP -> submitting up to $SLOTS"
  if [ "$SLOTS" -le 0 ]; then echo "[capacity] at cap, skipping this sweep"; sleep 1800; continue; fi
  python -m hpc.launch --job_type eval_listener \
    --cluster-config tacc --preset v2 \
    --datasets $SCRATCH/data/datasets/dev_set_v2 \
    --require-priority-list --priority-file $LIST \
    --config-yaml dcagent_eval_config_no_override.yaml \
    --force-reeval --once --verbose --slurm-time 47:59:00 --max-jobs-submitted "$SLOTS" 2>&1 | grep -iE "Submitted batch|-> Submitted|No eligible|error"
  sleep 1800
done
