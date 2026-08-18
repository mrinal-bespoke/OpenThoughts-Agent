#!/bin/bash
# Flag (and optionally cancel) legs that are RUNNING but writing nothing.
# squeue cannot see this: a wedged harbor run stays "RUNNING" until the wall kills it and takes
# every scored trial with it. Detect via file mtime in the run dir. Scored trials survive on disk,
# so cancelling a stalled leg is strictly better than letting it burn to the wall -- the next warm
# resume picks up where it stopped.
KILL="${1:-no}"
for j in $(squeue -u "$USER" -h -t R -o "%i"); do
  L=$(ls -t "$SCRATCH"/OpenThoughts-Agent/eval/tacc/logs/*_"${j}".out 2>/dev/null | head -1)
  RD=$(grep -m1 -oE "Run dir: .*" "$L" 2>/dev/null | awk '{print $3}')
  [ -n "$RD" ] && [ -d "$RD" ] || continue
  EL=$(squeue -j "$j" -h -o "%M")
  case "$EL" in *-*|*:*:*) ;; *) continue ;; esac      # only judge legs running >1h
  T=$(find "$RD" -maxdepth 2 -newermt "-45 minutes" 2>/dev/null | wc -l)
  if [ "$T" -eq 0 ]; then
    S=$(find "$RD" -maxdepth 2 -name "result*.json" 2>/dev/null | grep -c "__")
    echo "[stall] job=$j elapsed=$EL scored=$S no writes in 45min -> $(basename "$RD")"
    [ "$KILL" = "kill" ] && { scancel "$j"; echo "[stall] cancelled $j (scored trials preserved for warm resume)"; }
  fi
done
