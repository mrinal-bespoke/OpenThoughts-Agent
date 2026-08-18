#!/bin/bash
# DeepSWE integration spike — oracle agent, no GPU, no model.
# Question: does our harbor (ad6e612d) run Datacurve DeepSWE tasks end-to-end,
# including the [verifier] environment_mode="separate" contract?
set -uo pipefail
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor/src
set -a; source $SCRATCH/keys.env; set +a
cd $SCRATCH/spike

TASKS="abs-module-cache-flags arktype-json-schema-refs-dependencies boa-hierarchical-evaluation-cancellation"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$SCRATCH/spike/jobs
mkdir -p "$OUT"

echo "=== harbor: $(cd $SCRATCH/harbor && git rev-parse --short HEAD) ==="
echo "=== tasks: $TASKS ==="
ARGS=""
for t in $TASKS; do ARGS="$ARGS -t $t"; done

set -x
harbor jobs start \
  -p "$SCRATCH/deep-swe/tasks" \
  $ARGS \
  --n-concurrent 3 \
  --agent oracle \
  --env daytona \
  --n-attempts 1 \
  --job-name "deepswe_spike_$TS" \
  --jobs-dir "$OUT"
set +x
echo "=== EXIT: $? ==="
echo "=== RESULT ==="
R="$OUT/deepswe_spike_$TS"
[ -f "$R/result.json" ] && python -c "
import json;d=json.load(open('$R/result.json'))
s=d.get('stats',{})
print('  trials:',d.get('n_total_trials'),'completed:',s.get('n_completed_trials'),'errored:',s.get('n_errored_trials'))
for k,v in (s.get('evals') or {}).items(): print('  ',k,'->',v.get('metrics'))
"
for d in "$R"/*__*/; do
  n=$(basename "$d")
  rj=$(ls "$d"/verifier/reward.json 2>/dev/null | head -1)
  echo "  $n : reward.json=$([ -n "$rj" ] && echo yes || echo NO)"
done
