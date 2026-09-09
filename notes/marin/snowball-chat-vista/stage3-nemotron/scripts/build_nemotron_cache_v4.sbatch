#!/bin/bash
# Build the Stage 3 (Nemotron Terminal) packed cache.
#
# This stage does NOT share the Chat/Thinking contract:
#   - turns column is `conversations`, not `conversation`/`messages`
#   - template is MARIN_CHAT_TEMPLATE (414d8559...), NOT Delphi V0 (7b2363db...)
#   - the budget is a mandated 1,888 steps, not a derived epoch
#
# v4 vs v3: v3 passed `$SNAP/**/*.parquet` to a glob.glob() call that lacked
# recursive=True, so `**` collapsed to `*` and matched only the 3 shards at
# depth 1 (dataset_adapters/). The 26 nested synthetic_tasks/ shards -- 139,841
# of 366,154 rows -- were silently dropped, while a `find`-based gate in this
# same script counted 29 and passed. Job 983290 ran 1h34m on 61.8% of the data.
# The list is now EXPLICIT, built from the frozen upstream pinning and verified
# three ways (upstream == recorded hashes == on-disk) by build_list.py.
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144   # gg is OverSubscribe=NO: whole node either way
#SBATCH --time=04:00:00
#SBATCH --job-name=nemotron-terminal-cache-v4

set -euo pipefail
S=/scratch/11694/mkumar73
REV=a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9
LIST=$S/tmp/nemotron_stage3_paths.txt
CACHE=$S/experiments/snowball-chat-vista/nemotron-terminal-cache-v4
TOK=$S/hub/hub/models--laion--snowball-67b-a2b-cooldown-step105149/snapshots/b4d2656de8c39b699f7bbcbc878ca96996b319f3

module purge; module load gcc/13.2.0
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTHONPATH=$S/marin:$S/marin/lib/levanter/src:$S/marin/lib/haliax/src:$S/marin/lib/marin/src:$S/marin/lib/rigging/src:$S/marin/lib/fray/src:$S/marin/lib/zephyr/src:$S/marin/lib/iris/src
cd $S/marin

# Gate on the LIST, not on a directory walk -- the walk is what lied last time.
N=$(grep -vc '^\s*\(#.*\)\?$' "$LIST")
echo "NEMOTRON_CACHE revision=$REV list=$LIST entries=$N (expect 29)"
[ "$N" -eq 29 ] || { echo "FATAL: list has $N entries, expected 29" >&2; exit 2; }
MISSING=$(while read -r p; do case "$p" in ''|\#*) continue;; esac; [ -f "$p" ] || echo "$p"; done < "$LIST")
[ -z "$MISSING" ] || { echo "FATAL: listed shards missing:" >&2; echo "$MISSING" >&2; exit 2; }
[ -e "$CACHE" ] && { echo "FATAL: $CACHE already exists; refusing to reuse a cache path" >&2; exit 2; }

$S/envs/marin-grug-sft/bin/python -m experiments.june_tpu_67b_a2b.moe.vista_snowball_chat \
  prepare-data \
    --parquet-list "$LIST" \
    --expect-files 29 \
    --cache-path "$CACHE" \
    --tokenizer-path "$TOK" \
    --stage nemotron_terminal
echo "NEMOTRON_CACHE_BUILD_DONE"
