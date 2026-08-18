#!/bin/bash
# Early failure detection, run at the top of every sweep. Prints a one-screen verdict so a
# problem is visible in the log instead of surfacing hours later as "nothing completed".
LOGD=$SCRATCH/OpenThoughts-Agent/eval/tacc/logs
echo "[health] $(date '+%F %H:%M') --------------------------------------------"
R=$(squeue -u "$USER" -h -t R | wc -l); P=$(squeue -u "$USER" -h -t PD | wc -l)
echo "[health] queue: running=$R pending=$P"
# 1) STALLED legs: running but no file written in 30 min (invisible to squeue)
st=0
for j in $(squeue -u "$USER" -h -t R -o "%i"); do
  L=$(ls -t $LOGD/*_${j}.out 2>/dev/null | head -1); [ -n "$L" ] || continue
  RD=$(grep -m1 -oE "Run dir: .*" "$L" 2>/dev/null | awk '{print $3}'); [ -d "$RD" ] || continue
  n=$(find "$RD" -maxdepth 2 -newermt "-30 minutes" 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && { echo "[health] STALLED $j $(basename $RD | cut -c1-40)"; st=$((st+1)); }
done
echo "[health] stalled legs: $st"
# 2) wall risk: <2h remaining and not near done
for j in $(squeue -u "$USER" -h -t R -o "%i"); do
  LE=$(squeue -j "$j" -h -o "%L"); case "$LE" in 0:*|1:*) echo "[health] WALL-RISK $j left=$LE";; esac
done
# 3) failure RATE in currently-running legs (occurrences per leg, not logs-that-mention)
for j in $(squeue -u "$USER" -h -t R -o "%i"); do
  L=$(ls -t $LOGD/*_${j}.out 2>/dev/null | head -1); [ -n "$L" ] || continue
  RD=$(grep -m1 -oE "Run dir: .*" "$L" 2>/dev/null | awk '{print $3}')
  tot=$(ls -d "$RD"/*__* 2>/dev/null | wc -l); [ "$tot" -gt 0 ] || continue
  e=$(grep -c "EnvironmentStartTimeout" "$L" 2>/dev/null)
  pct=$(( e * 100 / tot ))
  [ "$pct" -ge 10 ] && echo "[health] HIGH-ENV-FAIL $j ${e}/${tot} (${pct}%)"
done
echo "[health] wall-timeouts in last 20 logs: $(for LG in $(ls -t $LOGD/*.out 2>/dev/null | head -20); do grep -l "DUE TO TIME LIMIT" "$LG" 2>/dev/null; done | wc -l)"
