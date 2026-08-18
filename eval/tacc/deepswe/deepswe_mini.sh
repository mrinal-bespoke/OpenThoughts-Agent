#!/bin/bash
# DeepSWE mini spike — 3 tasks (python/go/typescript), storage patched 20GB->10GB.
# Q1: does the 10GB cap actually break these builds, or is 20GB over-declared?
# Q2: if they build, does the separate-verifier contract produce reward.json/ctrf.json?
set -uo pipefail
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor/src
set -a; source $SCRATCH/keys.env; set +a
cd $SCRATCH/spike
TS=$(date +%Y%m%d_%H%M%S)
OUT=$SCRATCH/spike/jobs
RUN="deepswe_mini_$TS"
echo "=== harbor $(cd $SCRATCH/harbor && git rev-parse --short HEAD) | run $RUN ==="
# 55-min hard wall so this can never run away like the 113-task attempt did
timeout 3300 harbor jobs start \
  -p "$SCRATCH/spike/mini_dataset" \
  --n-concurrent 3 --agent oracle --env daytona --n-attempts 1 \
  --job-name "$RUN" --jobs-dir "$OUT"
echo "=== harbor exit: $? ==="
R="$OUT/$RUN"
echo "=== RESULT ==="
[ -f "$R/result.json" ] && python -c "
import json;d=json.load(open('$R/result.json'));s=d.get('stats',{})
print('  total',d.get('n_total_trials'),'completed',s.get('n_completed_trials'),'errored',s.get('n_errored_trials'))
for k,v in (s.get('evals') or {}).items(): print('  ',k,'->',v.get('metrics'))
"
echo "=== verifier artifacts (the actual question) ==="
echo "  reward.json: $(find $R -name reward.json 2>/dev/null | wc -l) / 3"
echo "  ctrf.json  : $(find $R -name ctrf.json 2>/dev/null | wc -l) / 3"
for d in "$R"/*__*/; do
  [ -d "$d" ] || continue
  echo "  $(basename $d): reward=$(find "$d" -name reward.json 2>/dev/null | wc -l) ctrf=$(find "$d" -name ctrf.json 2>/dev/null | wc -l)"
done
echo "=== DONE $(date +%H:%M:%S) ==="
