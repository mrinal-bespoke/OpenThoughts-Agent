#!/bin/bash
# Remove models that ALREADY have a running/pending job from the priority list.
#
# WHY: --force-reeval is required for flawed_summ (every target has an old Finished row), but it
# also bypasses the listener's active-pair guard (unified_eval_listener.py: the `(hf_model,dataset)
# in active_pairs` check is skipped when force_reeval is set). So each sweep happily re-submits a
# model that is mid-flight. Observed 2026-08-06: 14 distinct models occupying 19 slots
# (crosscodeeval_typescript x3, issue_tasks x3) — ~30% of a capped 16-slot budget wasted on dupes.
L="$1"
[ -f "$L" ] || exit 0
squeue -u "$USER" -h -o "%.200j" \
  | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/^(eval|res)_//; s/_(tb2|v2|bench_2|set_v2)$//' \
  | sort -u > /tmp/inflight.$$
before=$(wc -l < "$L")
awk -F/ '{print $NF"\t"$0}' "$L" | grep -v -F -f <(sed 's/^/^/;s/$/\t/' /tmp/inflight.$$ 2>/dev/null || echo "__none__") 2>/dev/null | cut -f2- > "$L.tmp" || cp "$L" "$L.tmp"
# safer: explicit filter
: > "$L.tmp"
while IFS= read -r m; do
  stub="${m##*/}"
  # unified_eval_listener uses the first 28 model-name characters while a job
  # is PENDING, then eval_harbor.sbatch renames it to the full model name once
  # it starts. Match both forms. A 28-char collision conservatively delays the
  # second model until the first starts; that is preferable to duplicate evals.
  pending_stub="${stub:0:28}"
  if grep -qxF "$stub" /tmp/inflight.$$ || grep -qxF "$pending_stub" /tmp/inflight.$$; then
    continue
  fi
  echo "$m" >> "$L.tmp"
done < "$L"
mv "$L.tmp" "$L"
rm -f /tmp/inflight.$$
echo "[inflight-filter] $before -> $(wc -l < "$L") (dropped models already in the queue)"
