#!/bin/bash
# Backfill hf_traces_link for legs that registered but whose HF upload 403'd against DCAgent2/.
# Runs detached: each upload pushes a full trace set and the login node reaps long processes.
cd $SCRATCH/OpenThoughts-Agent
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor/src:$SCRATCH/OpenThoughts-Agent
set -a; source $SCRATCH/keys.env; set +a
for D in \
  dev_set_v2_a1_crosscodeeval_python_20260805_125426 \
  dev_set_v2_a1_crosscodeeval_typescript_20260805_125428 \
  dev_set_v2_a1_exercism_python_20260805_125430 \
  dev_set_v2_a1_issue_tasks_20260805_125431 \
  dev_set_v2_a1_magicoder_20260805_125433 \
  terminal_bench_2_a1_softwareheritage_20260803_154955 \
  terminal_bench_2_a1_stack_selfdoc_20260803_154957 \
  terminal_bench_2_a1_synatra_20260804_134728 ; do
  P=$SCRATCH/eval_jobs/$D
  case "$D" in dev_set_v2_*) B=dev_set_v2; S2=${D#dev_set_v2_};; *) B=terminal_bench_2; S2=${D#terminal_bench_2_};; esac
  STUB=$(echo "$S2" | sed -E 's/_[0-9]{8}_[0-9]{6}$//' | sed 's/^a1_/a1-/')
  echo "=========== $STUB ($B) ==========="
  python scripts/database/manual_db_eval_push.py \
    --job-dir "$P" --model-name "DCAgent/$STUB" --benchmark-name "$B" \
    --agent-name terminus-2 --username mkumar73 \
    --hf-repo "laion/$D" --hf-episodes last \
    --error-mode skip_on_error --forced-update 2>&1 | tail -6
done
echo "=== BACKFILL DONE $(date +%H:%M:%S) ==="
