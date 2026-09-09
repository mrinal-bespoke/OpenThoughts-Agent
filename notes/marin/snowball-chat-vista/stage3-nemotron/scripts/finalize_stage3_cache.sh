#!/bin/bash
# Finalize the Stage 3 cache once 983710 completes:
#   1. prove the build actually finished and every shard is committed
#   2. write the provenance sidecar from the SAME explicit list the build used
#   3. validate it, which now also enforces source_files=29
#   4. report the exact StageSpec values to replace the placeholder zeros
set -euo pipefail
S=/scratch/11694/mkumar73
CACHE=$S/experiments/snowball-chat-vista/nemotron-terminal-cache-v4
LIST=$S/tmp/nemotron_stage3_paths.txt
LOG=$S/slurm-983710.out
TOK=$S/hub/hub/models--laion--snowball-67b-a2b-cooldown-step105149/snapshots/b4d2656de8c39b699f7bbcbc878ca96996b319f3
PY=$S/envs/marin-grug-sft/bin/python
export PYTHONPATH=$S/marin:$S/marin/lib/levanter/src:$S/marin/lib/haliax/src:$S/marin/lib/marin/src:$S/marin/lib/rigging/src:$S/marin/lib/fray/src:$S/marin/lib/zephyr/src:$S/marin/lib/iris/src
cd $S/marin

echo "=== 1. build completion ==="
grep -q NEMOTRON_CACHE_BUILD_DONE "$LOG" || { echo "FATAL: no BUILD_DONE marker in $LOG" >&2; exit 2; }
sacct -j 983710 --format=JobID,State,Elapsed -n | head -2
grep -c "source_shard:" "$LOG" | sed 's/^/resolved source shards: /'

echo "=== 2. cache integrity ==="
TMP=$(find "$CACHE" -name "*.tmp.*" | wc -l)
[ "$TMP" -eq 0 ] || { echo "FATAL: $TMP unfinalized .tmp shards remain" >&2; exit 2; }
ls "$CACHE/train" | grep -c '^part-' | sed 's/^/committed shards: /'
[ -f "$CACHE/train/.stats.json" ] || { echo "FATAL: no train/.stats.json" >&2; exit 2; }

echo "=== 3. write provenance (same explicit list as the build) ==="
$PY -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat write-provenance \
  --cache-path "$CACHE" --stage nemotron_terminal \
  --dataset-id nvidia/Nemotron-Terminal-Corpus \
  --dataset-revision a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9 \
  --parquet-list "$LIST" --expect-files 29 --tokenizer-path "$TOK"

echo "=== 4. validate provenance (exercises source_files=29) ==="
$PY - <<PY
import json, sys
sys.path.insert(0, "$S/marin")
from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import validate_cache_provenance, STAGES
rec = validate_cache_provenance("$CACHE", "nemotron_terminal", "$TOK")
print(f"  provenance OK: {len(rec['shards'])} shards == pinned {STAGES['nemotron_terminal'].source_files}")
st = json.load(open("$CACHE/train/.stats.json"))
tokens = st.get("total_tokens"); examples = st.get("total_elements")
import os
shards = len([d for d in os.listdir("$CACHE/train") if d.startswith("part-")])
print()
print("  === replace the placeholder zeros in STAGES['nemotron_terminal'] with: ===")
print(f"      cache_tokens={tokens},")
print(f"      cache_examples={examples},")
print(f"      cache_shards={shards},")
print()
print(f"  sanity: examples vs pinned corpus rows 366,154 -> {'MATCH' if examples == 366154 else 'DIFFERS (packing may merge docs -- check before trusting)'}")
PY
