#!/bin/bash
# GATE: current-format DeepSWE (v1.1, [[verifier.collect]]) with the collect-hook patch.
# Oracle applies each task's own reference solution -> reward MUST be 1.
# storage: task declares 20480; Daytona caps at 10240 -> runtime --override-storage-mb,
# so task semantics stay byte-identical to upstream (no private dialect).
set -uo pipefail
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor_test/src      # PATCHED copy; live campaign untouched
set -a; source $SCRATCH/keys.env; set +a
cd $SCRATCH/spike
TS=$(date +%Y%m%d_%H%M%S); RUN="deepswe_v11_gate_$TS"
echo "=== harbor $(cd $SCRATCH/harbor_test && git rev-parse --short HEAD) + collect-hook patch ==="
timeout 3000 harbor jobs start \
  -p "$SCRATCH/spike/deepswe_v11" \
  --n-concurrent 3 --agent oracle --env daytona --n-attempts 1 \
  --override-storage-mb 10240 \
  --job-name "$RUN" --jobs-dir "$SCRATCH/spike/jobs" 2>&1 | tail -10
R="$SCRATCH/spike/jobs/$RUN"
echo "=== GATE RESULT ==="
python - "$R" <<'PY'
import json,sys,glob,os
R=sys.argv[1]
d=json.load(open(f"{R}/result.json")); s=d.get("stats",{}) or {}
print(f"  trials {s.get('n_completed_trials')}/{d.get('n_total_trials')} errored={s.get('n_errored_trials')}")
for k,v in (s.get("evals") or {}).items():
    for m in (v.get("metrics") or []): print(f"  metrics: {m}")
pats=glob.glob(R+"/*__*/artifacts/model.patch")
print(f"  model.patch collected: {len(pats)}/3")
for f in sorted(pats): print(f"    {os.path.getsize(f):>7} bytes  {os.path.basename(os.path.dirname(os.path.dirname(f)))[:32]}")
PY
echo "=== GATE DONE $(date +%H:%M:%S) ==="
