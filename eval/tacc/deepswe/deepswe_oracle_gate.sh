#!/bin/bash
# DECISIVE GATE for the pre_artifacts hook.
# Same 3 tasks, same oracle agent as the 2026-08-03 spike which scored f2p=0.0 / p2p=1.0
# (env + verifier healthy, submission empty because model.patch was never created).
# PASS = f2p rises to ~1.0. Anything else means the hook runs but does not produce a
# scoring submission, which is worse than not shipping it.
set -uo pipefail
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor_deepswe/src        # PATCHED copy, not the live campaign's
set -a; source $SCRATCH/keys.env; set +a
cd $SCRATCH/spike
TS=$(date +%Y%m%d_%H%M%S); RUN="deepswe_gate_$TS"
echo "=== harbor: $(cd $SCRATCH/harbor_deepswe && git rev-parse --short HEAD) + pre_artifacts patch ==="
timeout 3000 harbor jobs start \
  -p "$SCRATCH/spike/mini_dataset" \
  --n-concurrent 3 --agent oracle --env daytona --n-attempts 1 \
  --job-name "$RUN" --jobs-dir "$SCRATCH/spike/jobs" 2>&1 | tail -12
R="$SCRATCH/spike/jobs/$RUN"
echo "=== GATE RESULT ==="
python - "$R" <<'PY'
import json,sys,glob,os
R=sys.argv[1]
d=json.load(open(f"{R}/result.json"))
s=d.get("stats",{}) or {}
print(f"  trials {s.get('n_completed_trials')}/{d.get('n_total_trials')} errored={s.get('n_errored_trials')}")
for k,v in (s.get("evals") or {}).items():
    for m in (v.get("metrics") or []):
        print(f"  metrics: {m}")
print(f"  model.patch produced: {len(glob.glob(R+'/*__*/artifacts/model.patch'))}/3")
for f in sorted(glob.glob(R+"/*__*/artifacts/model.patch")):
    print(f"    {os.path.basename(os.path.dirname(os.path.dirname(f)))[:34]}  {os.path.getsize(f)} bytes")
PY
echo "=== GATE DONE $(date +%H:%M:%S) ==="
