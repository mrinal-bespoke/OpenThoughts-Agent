#!/bin/bash
# Remove trial dirs that have no config.json.
#
# WHY THIS MUST RUN EVERY SWEEP, not once: any job that dies mid-flight (scancel, SLURM wall,
# artifact-flush timeout) leaves trial dirs whose config.json was never written. Harbor's warm
# resume then opens every trial dir's config.json unconditionally and aborts the whole leg with
# FileNotFoundError ~3 min in. One stale dir kills the entire re-fire, and the loop will happily
# retry it forever -- that cost 11 sweeps / ~66 failed jobs / 5h of zero progress on 2026-08-05.
# A dir with no config.json carries no resumable state, so deleting it just forces a fresh trial.
n=0
for d in "$SCRATCH"/eval_jobs/*/; do
  [ -d "$d" ] || continue
  while IFS= read -r t; do
    [ -f "$t/config.json" ] || { rm -rf "$t"; n=$((n+1)); }
  done < <(find "$d" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)
done
echo "[prune] removed $n partial trial dir(s)"
