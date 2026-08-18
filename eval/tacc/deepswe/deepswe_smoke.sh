#!/bin/bash
# DAYTONA SMOKE: both halves of the contract, on real sandboxes.
#  A) SUCCESS  -> hook runs, model.patch transfers, oracle reward 1.0
#  B) FAILURE  -> hook exits 17 -> trial must fail CLOSED (VerifierCollectFailedError,
#                 unscored) and must NOT record a 0 that looks like a real model failure.
set -uo pipefail
export PATH=/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH
export PYTHONPATH=$SCRATCH/harbor_test/src
set -a; source $SCRATCH/keys.env; set +a
cd $SCRATCH/spike
TS=$(date +%Y%m%d_%H%M%S)
echo "=== harbor $(cd $SCRATCH/harbor_test && git rev-parse --short HEAD) + collect patch v7 ==="

run () { # $1=label $2=taskdir
  RUN="deepswe_smoke_$1_$TS"
  timeout 2400 harbor jobs start -p "$2" --n-concurrent 1 --agent oracle --env daytona \
    --n-attempts 1 --override-storage-mb 10240 --job-name "$RUN" --jobs-dir "$SCRATCH/spike/jobs" >/dev/null 2>&1
  R="$SCRATCH/spike/jobs/$RUN"
  echo "--- $1 ---"
  python - "$R" <<'PY'
import json,sys,glob,os
R=sys.argv[1]
try: d=json.load(open(f"{R}/result.json"))
except Exception as e: print("  no result.json:",e); raise SystemExit
s=d.get("stats",{}) or {}
print(f"  completed={s.get('n_completed_trials')} errored={s.get('n_errored_trials')}")
for k,v in (s.get("evals") or {}).items():
    print(f"  metrics: {v.get('metrics')}")
    ex=v.get("exception_stats") or {}
    if ex: print(f"  exception_stats: {ex}")
print(f"  model.patch: {len(glob.glob(R+'/*__*/artifacts/model.patch'))}")
for f in glob.glob(R+"/*__*/result.json"):
    t=json.load(open(f))
    ei=t.get("exception_info") or {}
    print(f"  trial: verifier_result={'set' if t.get('verifier_result') else 'None'} exception={ei.get('exception_type')}")
PY
}
run success  "$SCRATCH/spike/deepswe_v11/abs-module-cache-flags"
run failhook "$SCRATCH/spike/deepswe_failhook/abs-module-cache-flags"
echo "=== SMOKE DONE $(date +%H:%M:%S) ==="
