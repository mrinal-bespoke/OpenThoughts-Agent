#!/bin/bash
#SBATCH --account=CCR24067
#SBATCH --partition=gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
# Keep this aligned with Vista's maintained node_exclusion_list in hpc/hpc.py.
# These nodes/racks have produced confirmed NODE_FAILs or repeat NCCL wedges.
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c636-121,c635-101,c641-061,c611-051,c636-152,c608-042,c634-142,c641-012,c634-052,c621-102,c637-042,c638-101,c638-102,c638-112,c639-[001-002,011-012,021-022,031-032,041-042,051-052,061-062,071-072,081-082,091-092,101-102,111-112,121-122,131-132,141-142,151-152]
# AIME24 runs ten sequential long-generation repetitions. The reviewed serving
# path needs roughly 9.5 hours for a fresh run, so a four-hour smoke default can
# terminate an otherwise healthy policy evaluation before it writes results.
#SBATCH --time=12:00:00
#SBATCH --job-name=grug-vllm-smoke
#SBATCH --output=/scratch/11694/mkumar73/logs/%x.%j.log

# Two-node Vista smoke test for the repaired Grug/Snowball HF export.
# The established default is TP=1 with experts sharded across two DP ranks.
# Parallel geometry is configurable so diagnostics can remove data-parallel
# coordination while still distributing model weights with pipeline parallelism.

set -euo pipefail

MODEL_REPO_EXPLICIT="${MODEL_REPO+x}"
MODEL_REVISION_EXPLICIT="${MODEL_REVISION+x}"
MODEL_REPO="${MODEL_REPO:-laion/grug-67b-a2b-sft-s3-agentic-step1903-repaired}"
MODEL_REVISION="${MODEL_REVISION:-0cb5c3491df5d8ae33ca5b1be513629a06c68882}"
EVAL_ENV="${EVAL_ENV:-$SCRATCH/envs/grug-eval}"
EVALCHEMY_ENV="${EVALCHEMY_ENV:-$SCRATCH/envs/grug-evalchemy}"
REPO_ROOT="${REPO_ROOT:-$SCRATCH/OpenThoughts-Agent}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
EVAL_NUM_CONCURRENT="${EVAL_NUM_CONCURRENT:-4}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_RESUME_MODE="${EVAL_RESUME_MODE:-force-fresh}"
VLLM_PYTHON_OVERLAY="${VLLM_PYTHON_OVERLAY:-}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-auto}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
RUN_EVAL="${RUN_EVAL:-0}"
RUN_AGENTIC="${RUN_AGENTIC:-0}"
RUN_CANARY="${RUN_CANARY:-0}"
CANARY_CONCURRENCY="${CANARY_CONCURRENCY:-4}"
CANARY_CALIBRATION_TOKENS="${CANARY_CALIBRATION_TOKENS:-512}"
CANARY_SUSTAINED_TOKENS="${CANARY_SUSTAINED_TOKENS:-4096}"
CANARY_DEPTH_TOKENS="${CANARY_DEPTH_TOKENS:-16384}"
CANARY_MIN_DEPTH_TOKENS="${CANARY_MIN_DEPTH_TOKENS:-8192}"
CANARY_CONTEXT_MARGIN="${CANARY_CONTEXT_MARGIN:-512}"
CANARY_METRICS_INTERVAL="${CANARY_METRICS_INTERVAL:-30}"
CANARY_STALL_TIMEOUT="${CANARY_STALL_TIMEOUT:-600}"
CANARY_UNHEALTHY_WINDOW="${CANARY_UNHEALTHY_WINDOW:-300}"
CANARY_MAX_FAILURE_RATE="${CANARY_MAX_FAILURE_RATE:-0.05}"
CANARY_MIN_HEALTH_FAILURES="${CANARY_MIN_HEALTH_FAILURES:-2}"
CANARY_HARD_TIMEOUT="${CANARY_HARD_TIMEOUT:-6000}"
CANARY_REQUEST_TIMEOUT="${CANARY_REQUEST_TIMEOUT:-900}"
CANARY_TIMEOUT_MULTIPLIER="${CANARY_TIMEOUT_MULTIPLIER:-1.0}"
CANARY_MIN_DURATION="${CANARY_MIN_DURATION:-2700}"
CANARY_SUSTAINED_SECONDS="${CANARY_SUSTAINED_SECONDS:-2400}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
if [ "$RUN_CANARY" = "1" ] && { [ "$RUN_EVAL" != "0" ] || [ "$RUN_AGENTIC" != "0" ]; }; then
  echo "FATAL: RUN_CANARY=1 requires RUN_EVAL=0 and RUN_AGENTIC=0" >&2
  exit 2
fi
if [ "$RUN_CANARY" = "1" ]; then
  RUN_KIND=canary
elif [ "$RUN_AGENTIC" = "1" ]; then
  RUN_KIND=agentic
elif [ "$RUN_EVAL" = "1" ]; then
  RUN_KIND=eval
else
  RUN_KIND=smoke
fi
if { [ "$RUN_EVAL" = "1" ] || [ "$RUN_CANARY" = "1" ]; } \
  && { [ -z "$MODEL_REPO_EXPLICIT" ] || [ -z "$MODEL_REVISION_EXPLICIT" ]; }; then
  echo "FATAL: eval and canary jobs must explicitly pin MODEL_REPO and MODEL_REVISION" >&2
  exit 2
fi
# RUN_DIR is overridable so a retry can reuse ONE result root. Derived from
# $SLURM_JOB_ID, every attempt lands in a fresh directory, so nothing can be
# resumed and a completed benchmark is paid for again. Default unchanged.
RUN_DIR="${RUN_DIR:-$SCRATCH/experiments/grug-nonagentic-eval/${RUN_KIND}-${SLURM_JOB_ID}}"

mkdir -p "$RUN_DIR" "$SCRATCH/logs" "$SCRATCH/hub"

module purge
module load gcc/14.2.0 cuda/12.8

# The private venv inherits shared Python packages, but venvs do not expose
# console scripts from their base environment. FlashInfer's sampler JIT calls
# `ninja` by name, so keep the shared environment's bin directory on PATH for
# every Ray worker.
export PATH="$EVAL_ENV/bin:/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH"
if [ -n "$VLLM_PYTHON_OVERLAY" ]; then
  test -f "$VLLM_PYTHON_OVERLAY/vllm/v1/engine/core.py"
  export PYTHONPATH="$VLLM_PYTHON_OVERLAY:$REPO_ROOT:${PYTHONPATH:-}"
else
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
fi
export HF_HOME="$SCRATCH/hub"
export HF_HUB_CACHE="$SCRATCH/hub"
export HF_HUB_OFFLINE=1
export NLTK_DATA="$SCRATCH/nltk_data"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
# Ray's log deduplicator can buffer the second DP actor's patch marker until
# shutdown because the two rank lines differ only by a number.  The marker is
# an integrity gate, so require immediate, lossless worker-log forwarding.
export RAY_DEDUP_LOGS=0
export CUDA_HOME="${TACC_CUDA_DIR}"
export TRITON_CC="$(command -v gcc)"
export CC="$TRITON_CC"
export CXX="$(command -v g++)"
export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export FI_EFA_FORK_SAFE=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export VLLM_USE_V2_MODEL_RUNNER=0
# The default FlashInfer sampler has repeatedly wedged this two-node DP+EP
# serving geometry in sample_tokens until the 7200s worker RPC timeout. Use
# vLLM's stable sampler path unless an explicit experiment opts back in.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_RAY_DP_PACK_STRATEGY=strict
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export VLLM_RINGBUFFER_WARNING_INTERVAL=600
# Compiled-kernel caches must be node-local. Sharing them over Vista's NFS can
# produce stale file handles when the two DP workers compile/read concurrently.
NODE_LOCAL_CACHE_ROOT="/tmp/${USER}/grug-eval-${SLURM_JOB_ID}"
export VLLM_CACHE_ROOT="$NODE_LOCAL_CACHE_ROOT/vllm"
export FLASHINFER_WORKSPACE_BASE="$NODE_LOCAL_CACHE_ROOT/flashinfer"
export TRITON_CACHE_DIR="$NODE_LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$NODE_LOCAL_CACHE_ROOT/inductor"
mkdir -p "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PYTHON="$EVAL_ENV/bin/python"
test -x "$PYTHON"
test -f "$REPO_ROOT/hpc/vllm/start_vllm_ray_controller.py"
if [ -n "$VLLM_PYTHON_OVERLAY" ]; then
  OVERLAY_MANIFEST="$VLLM_PYTHON_OVERLAY/manifest.json"
  test -f "$OVERLAY_MANIFEST"
  OVERLAY_SOURCE_PACKAGE=$("$PYTHON" - "$OVERLAY_MANIFEST" <<'PYEOF'
import json
import sys

print(json.load(open(sys.argv[1]))["source_package"])
PYEOF
)
  "$PYTHON" "$REPO_ROOT/hpc/vllm/build_vllm_dp_coordination_overlay.py" \
    --source-package "$OVERLAY_SOURCE_PACKAGE" \
    --overlay-root "$VLLM_PYTHON_OVERLAY"
fi

if ! [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || [ "$MAX_MODEL_LEN" -le 64 ]; then
  echo "FATAL: MAX_MODEL_LEN must be an integer greater than 64" >&2
  exit 2
fi
EVAL_MAX_LENGTH=$((MAX_MODEL_LEN - 64))
if [ "$RUN_EVAL" = "1" ] && [ -n "${NONAGENTIC_LEGS:-}" ]; then
  PLANNED_EVAL_ROOT="${EVAL_ROOT_OVERRIDE:-$RUN_DIR/results}"
  RESUME_PREFLIGHT_ARGS=(
    --eval-root "$PLANNED_EVAL_ROOT"
    --legs "$NONAGENTIC_LEGS"
    --expected-max-model-len "$EVAL_MAX_LENGTH"
    --resume-mode "$EVAL_RESUME_MODE"
  )
  if [ -n "$EVAL_LIMIT" ]; then
    RESUME_PREFLIGHT_ARGS+=(--limit "$EVAL_LIMIT")
  fi
  case "$EVAL_RESUME_MODE" in
    auto) RESUME_PREFLIGHT_ARGS+=(--require-fingerprint) ;;
    force-fresh|off) ;;
    *) echo "FATAL: EVAL_RESUME_MODE must be auto, force-fresh, or off" >&2; exit 2 ;;
  esac
  "$PYTHON" "$REPO_ROOT/hpc/vllm/validate_eval_resume_fingerprint.py" \
    "${RESUME_PREFLIGHT_ARGS[@]}"
fi

MODEL_PATH=$("$PYTHON" - "$MODEL_REPO" "$MODEL_REVISION" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
PYEOF
)

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_COUNT="${#NODES[@]}"
if [ "$NODE_COUNT" -lt 1 ]; then
  echo "FATAL: expected at least 1 node, got $NODE_COUNT" >&2
  exit 2
fi
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-$NODE_COUNT}"
if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$PIPELINE_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$DATA_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: tensor/pipeline/data parallel sizes must be positive integers" >&2
  exit 2
fi
if [ "$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE * DATA_PARALLEL_SIZE))" -ne "$NODE_COUNT" ]; then
  echo "FATAL: TP*PP*DP must equal allocated GPUs/nodes: ${TENSOR_PARALLEL_SIZE}*${PIPELINE_PARALLEL_SIZE}*${DATA_PARALLEL_SIZE} != ${NODE_COUNT}" >&2
  exit 2
fi
if [ "$ENABLE_EXPERT_PARALLEL" != "0" ] && [ "$ENABLE_EXPERT_PARALLEL" != "1" ]; then
  echo "FATAL: ENABLE_EXPERT_PARALLEL must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$EVAL_NUM_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: EVAL_NUM_CONCURRENT must be a positive integer" >&2
  exit 2
fi
if ! [[ "$CANARY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: CANARY_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: MAX_NUM_SEQS must be a positive integer" >&2
  exit 2
fi
for canary_value in \
  "$CANARY_CALIBRATION_TOKENS" \
  "$CANARY_SUSTAINED_TOKENS" \
  "$CANARY_DEPTH_TOKENS" \
  "$CANARY_MIN_DEPTH_TOKENS" \
  "$CANARY_CONTEXT_MARGIN"; do
  if ! [[ "$canary_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: canary token sizes and context margin must be positive integers" >&2
    exit 2
  fi
done
if [ "$RUN_CANARY" = "1" ]; then
  "$PYTHON" - \
    "$CANARY_METRICS_INTERVAL" \
    "$CANARY_STALL_TIMEOUT" \
    "$CANARY_UNHEALTHY_WINDOW" \
    "$CANARY_MAX_FAILURE_RATE" \
    "$CANARY_MIN_HEALTH_FAILURES" \
    "$CANARY_HARD_TIMEOUT" \
    "$CANARY_REQUEST_TIMEOUT" \
    "$CANARY_TIMEOUT_MULTIPLIER" \
    "$CANARY_MIN_DURATION" \
    "$CANARY_SUSTAINED_SECONDS" <<'PYEOF'
import math
import sys

(
    metrics_interval,
    stall_timeout,
    unhealthy_window,
    max_failure_rate,
    min_health_failures,
    hard_timeout,
    request_timeout,
    timeout_multiplier,
    min_duration,
    sustained_seconds,
) = sys.argv[1:]
positive_floats = {
    "CANARY_METRICS_INTERVAL": metrics_interval,
    "CANARY_STALL_TIMEOUT": stall_timeout,
    "CANARY_UNHEALTHY_WINDOW": unhealthy_window,
    "CANARY_HARD_TIMEOUT": hard_timeout,
    "CANARY_REQUEST_TIMEOUT": request_timeout,
    "CANARY_MIN_DURATION": min_duration,
    "CANARY_SUSTAINED_SECONDS": sustained_seconds,
}
for name, raw_value in positive_floats.items():
    try:
        value = float(raw_value)
    except ValueError:
        raise SystemExit(f"FATAL: {name} must be numeric") from None
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"FATAL: {name} must be finite and positive")
try:
    failure_rate = float(max_failure_rate)
except ValueError:
    raise SystemExit("FATAL: CANARY_MAX_FAILURE_RATE must be numeric") from None
if not math.isfinite(failure_rate) or not 0 < failure_rate <= 1:
    raise SystemExit("FATAL: CANARY_MAX_FAILURE_RATE must be in (0, 1]")
try:
    multiplier = float(timeout_multiplier)
except ValueError:
    raise SystemExit("FATAL: CANARY_TIMEOUT_MULTIPLIER must be numeric") from None
if not math.isfinite(multiplier) or multiplier < 1:
    raise SystemExit("FATAL: CANARY_TIMEOUT_MULTIPLIER must be at least 1")
if not min_health_failures.isdigit() or int(min_health_failures) <= 0:
    raise SystemExit("FATAL: CANARY_MIN_HEALTH_FAILURES must be a positive integer")
PYEOF
fi
if [ "$RUN_CANARY" = "1" ] && [ "$MAX_NUM_SEQS" -ne "$CANARY_CONCURRENCY" ]; then
  echo "FATAL: sustained canary requires MAX_NUM_SEQS == CANARY_CONCURRENCY" >&2
  exit 2
fi
if [ "$RUN_CANARY" = "1" ] \
  && [ "$MAX_MODEL_LEN" -lt "$((CANARY_DEPTH_TOKENS + CANARY_CONTEXT_MARGIN))" ]; then
  echo "FATAL: MAX_MODEL_LEN must fit CANARY_DEPTH_TOKENS plus CANARY_CONTEXT_MARGIN" >&2
  exit 2
fi
if [ -n "$EVAL_LIMIT" ] && ! [[ "$EVAL_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: EVAL_LIMIT must be empty or a positive integer" >&2
  exit 2
fi
HEAD_NODE="${NODES[0]}"
HEAD_IP=$(srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$HEAD_NODE" hostname -i | awk '{print $1}')
RAY_ADDRESS="$HEAD_IP:6379"
export RAY_ADDRESS VLLM_HOST_IP="$HEAD_IP"

DP_ARGS=()
if [ "$DATA_PARALLEL_SIZE" -gt 1 ]; then
  DP_ARGS=(
    --data-parallel-address "$HEAD_IP"
    --data-parallel-size-local 1
    --data-parallel-backend ray
  )
fi

EP_ARGS=()
if [ "$ENABLE_EXPERT_PARALLEL" = "1" ]; then
  EP_ARGS=(--enable-expert-parallel)
fi

ASYNC_SCHEDULING_ARGS=()
case "$ASYNC_SCHEDULING" in
  auto) ;;
  0) ASYNC_SCHEDULING_ARGS=(--no-async-scheduling) ;;
  1) ASYNC_SCHEDULING_ARGS=(--async-scheduling) ;;
  *) echo "FATAL: ASYNC_SCHEDULING must be auto, 0, or 1" >&2; exit 2 ;;
esac

THINKING_ARGS=()
if [ "$ENABLE_THINKING" = "1" ]; then
  # This checkpoint's shipped chat template uses this kwarg to emit the
  # training-time `Reasoning: /think` control header for chat requests.
  THINKING_ARGS=(--default-chat-template-kwargs '{"enable_thinking": true}')
fi

if [ "$RUN_EVAL" = "1" ]; then
  RUN_MANIFEST_ARGS=(
    --output "$RUN_DIR/run_manifest.json"
    --file "executed_wrapper=${BASH_SOURCE[0]}"
    --file "wrapper_source=$REPO_ROOT/hpc/vllm/smoke_grug_vllm_tacc.sbatch"
    --file "overlay_builder=$REPO_ROOT/hpc/vllm/build_vllm_dp_coordination_overlay.py"
    --file "resume_validator=$REPO_ROOT/hpc/vllm/validate_eval_resume_fingerprint.py"
    --file "result_validator=$REPO_ROOT/hpc/vllm/validate_eval_result.py"
    --file "manifest_writer=$REPO_ROOT/hpc/vllm/write_eval_run_manifest.py"
    --setting "MODEL_REPO=$MODEL_REPO"
    --setting "MODEL_REVISION=$MODEL_REVISION"
    --setting "MODEL_PATH=$MODEL_PATH"
    --setting "ENABLE_THINKING=$ENABLE_THINKING"
    --setting "NONAGENTIC_LEGS=${NONAGENTIC_LEGS:-<unset>}"
    --setting "EVAL_ROOT=${EVAL_ROOT_OVERRIDE:-$RUN_DIR/results}"
    --setting "EVAL_RESUME_MODE=$EVAL_RESUME_MODE"
    --setting "EVAL_LIMIT=${EVAL_LIMIT:-<unset>}"
    --setting "RUN_KIND=$RUN_KIND"
    --setting "MAX_MODEL_LEN=$MAX_MODEL_LEN"
    --setting "EVAL_MAX_LENGTH=$EVAL_MAX_LENGTH"
    --setting "MAX_NUM_SEQS=$MAX_NUM_SEQS"
    --setting "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
    --setting "EVAL_NUM_CONCURRENT=$EVAL_NUM_CONCURRENT"
    --setting "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
    --setting "PIPELINE_PARALLEL_SIZE=$PIPELINE_PARALLEL_SIZE"
    --setting "DATA_PARALLEL_SIZE=$DATA_PARALLEL_SIZE"
    --setting "ENABLE_EXPERT_PARALLEL=$ENABLE_EXPERT_PARALLEL"
    --setting "ASYNC_SCHEDULING=$ASYNC_SCHEDULING"
    --setting "MOE_BACKEND=$MOE_BACKEND"
    --setting "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
  )
  if [ -n "$VLLM_PYTHON_OVERLAY" ]; then
    RUN_MANIFEST_ARGS+=(--overlay-manifest "$VLLM_PYTHON_OVERLAY/manifest.json")
  fi
  "$PYTHON" "$REPO_ROOT/hpc/vllm/write_eval_run_manifest.py" "${RUN_MANIFEST_ARGS[@]}"
fi

AGENTIC_SERVE_ARGS=()
if [ "$RUN_AGENTIC" = "1" ]; then
  # Match Marin issue #7743's model.yaml. Terminus-2 uses the OpenAI tools API,
  # while distributed=true enables the fork's distributed checkpoint loader.
  AGENTIC_SERVE_ARGS=(
    --enable-auto-tool-choice
    --tool-call-parser hermes
    --model-loader-extra-config '{"distributed":true}'
  )
fi

cleanup() {
  set +e
  if [ -n "${VLLM_PID:-}" ]; then kill "$VLLM_PID" 2>/dev/null; fi
  for node in "${NODES[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$node" \
      "$PYTHON" -m ray.scripts.scripts stop --force >/dev/null 2>&1
  done
}
trap cleanup EXIT INT TERM

echo "nodes=${NODES[*]} head_ip=$HEAD_IP ray=$RAY_ADDRESS"
echo "model=$MODEL_PATH"
echo "parallel_geometry=tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-dp${DATA_PARALLEL_SIZE}-ep${ENABLE_EXPERT_PARALLEL} moe_backend=$MOE_BACKEND async_scheduling=$ASYNC_SCHEDULING eval_num_concurrent=$EVAL_NUM_CONCURRENT eval_limit=${EVAL_LIMIT:-full} vllm_overlay=${VLLM_PYTHON_OVERLAY:-none}"

for node in "${NODES[@]}"; do
  srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$node" \
    "$PYTHON" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
done

srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$HEAD_NODE" \
  "$PYTHON" -m ray.scripts.scripts start --head \
  --node-ip-address="$HEAD_IP" --port=6379 --num-gpus=1 --num-cpus=64 \
  --object-store-memory=10737418240 --block >"$RUN_DIR/ray-head.log" 2>&1 &
RAY_HEAD_PID=$!

sleep 10
RAY_WORKER_PIDS=()
for WORKER_NODE in "${NODES[@]:1}"; do
  WORKER_IP=$(srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$WORKER_NODE" hostname -i | awk '{print $1}')
  srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$WORKER_NODE" \
    "$PYTHON" -m ray.scripts.scripts start --address="$RAY_ADDRESS" \
    --node-ip-address="$WORKER_IP" --num-gpus=1 --num-cpus=64 \
    --object-store-memory=10737418240 --block >"$RUN_DIR/ray-worker-${WORKER_NODE}.log" 2>&1 &
  RAY_WORKER_PIDS+=("$!")
done

"$PYTHON" - "$RAY_ADDRESS" "$NODE_COUNT" <<'PYEOF'
import sys
import time
import ray

address = sys.argv[1]
expected_nodes = int(sys.argv[2])
deadline = time.time() + 600
while time.time() < deadline:
    try:
        ray.init(address=address, ignore_reinit_error=True)
        resources = ray.cluster_resources()
        live = [n for n in ray.nodes() if n.get("Alive")]
        if resources.get("GPU", 0) >= expected_nodes and len(live) >= expected_nodes:
            print(f"RAY_READY resources={resources} live_nodes={len(live)}")
            ray.shutdown()
            break
        ray.shutdown()
    except Exception as exc:
        print(f"waiting for Ray: {exc}")
    time.sleep(10)
else:
    raise SystemExit(f"FATAL: {expected_nodes}-node Ray cluster did not become ready")
PYEOF

"$PYTHON" "$REPO_ROOT/hpc/vllm/start_vllm_ray_controller.py" \
  --ray-address "$RAY_ADDRESS" \
  --host "$HEAD_IP" \
  --port "$API_PORT" \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_REPO" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  "${DP_ARGS[@]}" \
  "${EP_ARGS[@]}" \
  "${ASYNC_SCHEDULING_ARGS[@]}" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --moe-backend "$MOE_BACKEND" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  "${THINKING_ARGS[@]}" \
  "${AGENTIC_SERVE_ARGS[@]}" \
  --no-enable-log-requests >"$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

"$PYTHON" - "$HEAD_IP" "$API_PORT" "$MODEL_REPO" "$RUN_DIR" "$VLLM_PID" <<'PYEOF'
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

host, port, model, run_dir, vllm_pid = sys.argv[1:]
base = f"http://{host}:{port}"
deadline = time.time() + 7200
last_error = None
while time.time() < deadline:
    try:
        state = Path(f"/proc/{vllm_pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        state = None
    if state in (None, "Z"):
        raise SystemExit(f"FATAL: vLLM process {vllm_pid} exited before becoming healthy")
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as response:
            if response.status == 200:
                break
    except Exception as exc:
        last_error = exc
    time.sleep(15)
else:
    raise SystemExit(f"FATAL: vLLM did not become healthy: {last_error}")

payload = json.dumps(
    {
        "model": model,
        "prompt": "The capital of France is",
        "temperature": 0,
        "max_tokens": 16,
    }
).encode()
request = urllib.request.Request(
    base + "/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=1800) as response:
    result = json.load(response)

text = result["choices"][0]["text"]
if not text.strip():
    raise SystemExit(f"FATAL: empty completion: {result}")
output = Path(run_dir) / "completion.json"
output.write_text(json.dumps(result, indent=2))
print(f"completion={text!r}")
print(f"result={output}")
print("GRUG_VLLM_SMOKE_OK")
PYEOF

if [ -n "$VLLM_PYTHON_OVERLAY" ]; then
  "$PYTHON" "$REPO_ROOT/hpc/vllm/verify_vllm_dp_patch_markers.py" \
    --log-path "$RUN_DIR/vllm.log" \
    --expected-size "$DATA_PARALLEL_SIZE" \
    --vllm-pid "$VLLM_PID" \
    --timeout-sec 60
fi

if [ "$RUN_CANARY" = "1" ]; then
  test -f "$REPO_ROOT/hpc/vllm/sustained_decode_canary.py"
  "$PYTHON" - \
    "$RUN_DIR/canary-manifest.json" \
    "${BASH_SOURCE[0]}" \
    "$REPO_ROOT/hpc/vllm/sustained_decode_canary.py" \
    "$MODEL_REPO" \
    "$MODEL_REVISION" \
    "$MODEL_PATH" \
    "$TENSOR_PARALLEL_SIZE" \
    "$PIPELINE_PARALLEL_SIZE" \
    "$DATA_PARALLEL_SIZE" \
    "$ENABLE_EXPERT_PARALLEL" \
    "$MAX_MODEL_LEN" \
    "$MAX_NUM_SEQS" \
    "$MAX_NUM_BATCHED_TOKENS" \
    "$ASYNC_SCHEDULING" \
    "$ENABLE_THINKING" \
    "$VLLM_PYTHON_OVERLAY" \
    "${SLURM_JOB_ID:-<unset>}" \
    "$RUN_DIR" \
    "$CANARY_CONCURRENCY" \
    "$CANARY_CALIBRATION_TOKENS" \
    "$CANARY_SUSTAINED_TOKENS" \
    "$CANARY_DEPTH_TOKENS" \
    "$CANARY_MIN_DEPTH_TOKENS" \
    "$CANARY_METRICS_INTERVAL" \
    "$CANARY_STALL_TIMEOUT" \
    "$CANARY_UNHEALTHY_WINDOW" \
    "$CANARY_MAX_FAILURE_RATE" \
    "$CANARY_MIN_HEALTH_FAILURES" \
    "$CANARY_HARD_TIMEOUT" \
    "$CANARY_REQUEST_TIMEOUT" \
    "$CANARY_TIMEOUT_MULTIPLIER" \
    "$CANARY_MIN_DURATION" \
    "$CANARY_SUSTAINED_SECONDS" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

(
    output,
    wrapper,
    canary,
    model_repo,
    model_revision,
    model_path,
    tp,
    pp,
    dp,
    ep,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    async_scheduling,
    enable_thinking,
    vllm_python_overlay,
    slurm_job_id,
    run_dir,
    canary_concurrency,
    calibration_tokens,
    sustained_tokens,
    depth_tokens,
    min_depth_tokens,
    metrics_interval,
    stall_timeout,
    unhealthy_window,
    max_failure_rate,
    min_health_failures,
    hard_timeout,
    request_timeout,
    timeout_multiplier,
    min_duration,
    sustained_seconds,
) = sys.argv[1:]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

payload = {
    "model_repo": model_repo,
    "model_revision": model_revision,
    "model_path": model_path,
    "geometry": {"tp": int(tp), "pp": int(pp), "dp": int(dp), "ep": int(ep)},
    "max_model_len": int(max_model_len),
    "max_num_seqs": int(max_num_seqs),
    "max_num_batched_tokens": int(max_num_batched_tokens),
    "async_scheduling": async_scheduling,
    "enable_thinking": int(enable_thinking),
    "vllm_python_overlay": vllm_python_overlay or None,
    "slurm_job_id": slurm_job_id,
    "run_dir": run_dir,
    "canary": {
        "concurrency": int(canary_concurrency),
        "calibration_tokens": int(calibration_tokens),
        "sustained_tokens": int(sustained_tokens),
        "depth_tokens": int(depth_tokens),
        "min_depth_tokens": int(min_depth_tokens),
        "metrics_interval": float(metrics_interval),
        "stall_timeout": float(stall_timeout),
        "unhealthy_window": float(unhealthy_window),
        "max_failure_rate": float(max_failure_rate),
        "min_health_failures": int(min_health_failures),
        "hard_timeout": float(hard_timeout),
        "request_timeout": float(request_timeout),
        "timeout_multiplier": float(timeout_multiplier),
        "min_duration": float(min_duration),
        "sustained_seconds": float(sustained_seconds),
    },
    "files": {
        "wrapper": {"path": wrapper, "sha256": digest(wrapper)},
        "canary": {"path": canary, "sha256": digest(canary)},
    },
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PYEOF
  "$PYTHON" "$REPO_ROOT/hpc/vllm/sustained_decode_canary.py" \
    --base-url "http://${HEAD_IP}:${API_PORT}" \
    --model "$MODEL_REPO" \
    --run-dir "$RUN_DIR" \
    --vllm-log "$RUN_DIR/vllm.log" \
    --concurrency "$CANARY_CONCURRENCY" \
    --metrics-interval "$CANARY_METRICS_INTERVAL" \
    --stall-timeout "$CANARY_STALL_TIMEOUT" \
    --unhealthy-window "$CANARY_UNHEALTHY_WINDOW" \
    --max-failure-rate "$CANARY_MAX_FAILURE_RATE" \
    --min-health-failures "$CANARY_MIN_HEALTH_FAILURES" \
    --hard-timeout "$CANARY_HARD_TIMEOUT" \
    --request-timeout "$CANARY_REQUEST_TIMEOUT" \
    --timeout-multiplier "$CANARY_TIMEOUT_MULTIPLIER" \
    --min-duration "$CANARY_MIN_DURATION" \
    --sustained-seconds "$CANARY_SUSTAINED_SECONDS" \
    --calibration-tokens "$CANARY_CALIBRATION_TOKENS" \
    --sustained-tokens "$CANARY_SUSTAINED_TOKENS" \
    --depth-tokens "$CANARY_DEPTH_TOKENS" \
    --min-depth-tokens "$CANARY_MIN_DEPTH_TOKENS"
fi

if [ "$RUN_EVAL" = "1" ]; then
  test -x "$EVALCHEMY_ENV/bin/evalchemy"
  EVAL_ROOT="${EVAL_ROOT_OVERRIDE:-$RUN_DIR/results}"
  mkdir -p "$EVAL_ROOT"

  run_eval_task() {
    local task="$1"
    local max_tokens="$2"
    local output_name="$3"
    local model_adapter="$4"
    local endpoint="$5"
    local num_fewshot="$6"
    local unsafe_code="${7:-0}"
    local model_args="model=${MODEL_REPO},base_url=http://${HEAD_IP}:${API_PORT}${endpoint},tokenizer=${MODEL_PATH},tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=${EVAL_NUM_CONCURRENT},max_retries=8,timeout=1800,max_length=${EVAL_MAX_LENGTH}"
    local extra_args=()
    local chat_template_args=()
    local resume_args=()
    local limit_args=()
    if [ "$unsafe_code" = "1" ]; then
      extra_args+=(--confirm_run_unsafe_code)
    fi
    # Chat-style generation tasks need the checkpoint's chat template. Classic
    # lm-eval loglikelihood tasks pass plain/JsonChatStr prompts to the
    # completions adapter; applying a chat template there turns them into an
    # object that the tokenizer path cannot rstrip().
    if [ "$model_adapter" = "local-chat-completions" ]; then
      chat_template_args+=(--apply_chat_template)
    fi
    if [ -n "${EVAL_RESUME_MODE:-}" ]; then
      resume_args+=(--resume-mode "$EVAL_RESUME_MODE")
    fi
    if [ -n "$EVAL_LIMIT" ]; then
      limit_args+=(--limit "$EVAL_LIMIT")
    fi
    local task_output_dir="$EVAL_ROOT/$output_name"
    local artifact_snapshot="$RUN_DIR/${task}-${EVAL_SEED:-42}.artifacts-before.txt"
    local leg_manifest="$RUN_DIR/${task}-${EVAL_SEED:-42}.protocol.json"
    local apply_chat_template=0
    if [ "$model_adapter" = "local-chat-completions" ]; then
      apply_chat_template=1
    fi
    LEG_MANIFEST_ARGS=(
      --output "$leg_manifest"
      --file "executed_wrapper=${BASH_SOURCE[0]}"
      --file "result_validator=$REPO_ROOT/hpc/vllm/validate_eval_result.py"
      --setting "TASK=$task"
      --setting "SEED=${EVAL_SEED:-42}"
      --setting "MODEL_REPO=$MODEL_REPO"
      --setting "MODEL_REVISION=$MODEL_REVISION"
      --setting "MODEL_ADAPTER=$model_adapter"
      --setting "MAX_LENGTH=$EVAL_MAX_LENGTH"
      --setting "MAX_TOKENS=$max_tokens"
      --setting "NUM_FEWSHOT=$num_fewshot"
      --setting "APPLY_CHAT_TEMPLATE=$apply_chat_template"
      --setting "ENABLE_THINKING=$ENABLE_THINKING"
      --setting "GEN_KWARGS=temperature=0.7,top_p=1.0,repetition_penalty=1.1"
      --setting "EVAL_RESUME_MODE=$EVAL_RESUME_MODE"
      --setting "EVAL_LIMIT=${EVAL_LIMIT:-<unset>}"
    )
    if [ -n "$VLLM_PYTHON_OVERLAY" ]; then
      LEG_MANIFEST_ARGS+=(--overlay-manifest "$VLLM_PYTHON_OVERLAY/manifest.json")
    fi
    "$PYTHON" "$REPO_ROOT/hpc/vllm/write_eval_run_manifest.py" "${LEG_MANIFEST_ARGS[@]}"
    mkdir -p "$task_output_dir"
    find "$task_output_dir" -type f -name 'results_*.json' -print > "$artifact_snapshot"
    echo "EVAL_START task=$task max_tokens=$max_tokens model=$model_adapter fewshot=$num_fewshot seed=${EVAL_SEED:-42}"
    # Keep model loading pinned/offline above, but allow Evalchemy to fetch a
    # benchmark dataset that is not already in the shared HF cache. OT-Agent has
    # its own top-level database package, so PYTHONPATH must remain unset here.
    env -u PYTHONPATH HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
      "$EVALCHEMY_ENV/bin/evalchemy" \
      --model "$model_adapter" \
      --model_args "$model_args" \
      --tasks "$task" \
      --gen_kwargs "temperature=0.7,top_p=1.0,repetition_penalty=1.1" \
      --max_tokens "$max_tokens" \
      --output_path "$EVAL_ROOT/$output_name" \
      --log_samples \
      --verbosity INFO \
      --num_fewshot "$num_fewshot" \
      --batch_size 1 \
      --seed "${EVAL_SEED:-42}" \
      "${limit_args[@]}" \
      "${resume_args[@]}" \
      "${chat_template_args[@]}" \
      "${extra_args[@]}"

    # Evalchemy can log a task-level exception, leave an older artifact as the
    # newest file, or write null/wrong-task/partial results while exiting zero.
    # Require a fresh seed-matching artifact with a numeric score and the full
    # policy sample count for the requested task.
    "$PYTHON" "$REPO_ROOT/hpc/vllm/validate_eval_result.py" \
      --output-dir "$EVAL_ROOT/$output_name" \
      --task "$task" \
      --seed "${EVAL_SEED:-42}" \
      --artifact-snapshot "$artifact_snapshot" \
      --model-repo "$MODEL_REPO" \
      --model-revision "$MODEL_REVISION" \
      --model-adapter "$model_adapter" \
      --max-length "$EVAL_MAX_LENGTH" \
      --max-tokens "$max_tokens"
    echo "EVAL_DONE task=$task"
  }

  MATH_TOKENS=8192
  if [ "$MATH_TOKENS" -gt "$((MAX_MODEL_LEN - 1024))" ]; then
    MATH_TOKENS=$((MAX_MODEL_LEN - 1024))
  fi

  # Policy-profile work is expressed as explicit task@seed legs. Only names
  # in this fixed case statement can run; malformed or unknown input fails
  # closed instead of becoming an arbitrary evaluator argument.
  run_named_eval_leg() {
    local task="$1"
    local seed="$2"
    case "$task" in
      MATH500)        EVAL_SEED="$seed" run_eval_task MATH500        "$MATH_TOKENS" "math500/seed-${seed}"        local-chat-completions /v1/chat/completions 0 ;;
      AIME24)         EVAL_SEED="$seed" run_eval_task AIME24         "$MATH_TOKENS" "aime24/seed-${seed}"         local-chat-completions /v1/chat/completions 0 ;;
      HumanEvalPlus)  EVAL_SEED="$seed" run_eval_task HumanEvalPlus  1024          "humanevalplus/seed-${seed}"  local-chat-completions /v1/chat/completions 0 1 ;;
      MBPPPlus)       EVAL_SEED="$seed" run_eval_task MBPPPlus       1024          "mbppplus/seed-${seed}"       local-chat-completions /v1/chat/completions 0 1 ;;
      GPQADiamond)    EVAL_SEED="$seed" run_eval_task GPQADiamond    "$MATH_TOKENS" "gpqa_diamond/seed-${seed}"  local-chat-completions /v1/chat/completions 0 ;;
      gsm8k)          EVAL_SEED="$seed" run_eval_task gsm8k          512           "gsm8k/seed-${seed}"          local-chat-completions /v1/chat/completions 0 ;;
      mmlu)           EVAL_SEED="$seed" run_eval_task mmlu           256           "mmlu/seed-${seed}"           local-completions      /v1/completions      5 ;;
      hellaswag)      EVAL_SEED="$seed" run_eval_task hellaswag      256           "hellaswag/seed-${seed}"      local-completions      /v1/completions      10 ;;
      arc_challenge)  EVAL_SEED="$seed" run_eval_task arc_challenge  256           "arc_challenge/seed-${seed}"  local-completions      /v1/completions      25 ;;
      arc_easy)       EVAL_SEED="$seed" run_eval_task arc_easy       256           "arc_easy/seed-${seed}"       local-completions      /v1/completions      0 ;;
      piqa)           EVAL_SEED="$seed" run_eval_task piqa           256           "piqa/seed-${seed}"           local-completions      /v1/completions      0 ;;
      winogrande)     EVAL_SEED="$seed" run_eval_task winogrande     256           "winogrande/seed-${seed}"     local-completions      /v1/completions      5 ;;
      openbookqa)     EVAL_SEED="$seed" run_eval_task openbookqa     256           "openbookqa/seed-${seed}"     local-completions      /v1/completions      0 ;;
      boolq)          EVAL_SEED="$seed" run_eval_task boolq          256           "boolq/seed-${seed}"          local-completions      /v1/completions      0 ;;
      truthfulqa_mc2) EVAL_SEED="$seed" run_eval_task truthfulqa_mc2 256           "truthfulqa_mc2/seed-${seed}" local-completions      /v1/completions      0 ;;
      lambada_openai) EVAL_SEED="$seed" run_eval_task lambada_openai 256           "lambada_openai/seed-${seed}" local-completions      /v1/completions      0 ;;
      triviaqa)       EVAL_SEED="$seed" run_eval_task triviaqa       128           "triviaqa/seed-${seed}"       local-chat-completions /v1/chat/completions 5 ;;
      nq_open)        EVAL_SEED="$seed" run_eval_task nq_open        128           "nq_open/seed-${seed}"        local-chat-completions /v1/chat/completions 5 ;;
      drop)           EVAL_SEED="$seed" run_eval_task drop           256           "drop/seed-${seed}"           local-chat-completions /v1/chat/completions 3 ;;
      *) echo "FATAL: unsupported non-agentic task: $task" >&2; return 2 ;;
    esac
  }

  # One stable task directory per benchmark makes interruption recovery cheap:
  # the wrapper requests only tasks without a valid seed-matching result. The
  # old task-list path is retained for the in-flight full9 profile.
  if [ -n "${NONAGENTIC_LEGS:-}" ]; then
    IFS=';' read -r -a eval_legs <<< "$NONAGENTIC_LEGS"
    for leg in "${eval_legs[@]}"; do
      task="${leg%@*}"
      seed="${leg##*@}"
      if [ "$task" = "$seed" ] || ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "FATAL: malformed non-agentic leg: $leg" >&2
        exit 2
      fi
      run_named_eval_leg "$task" "$seed"
    done
  else
    if [ -n "${NONAGENTIC_TASKS_SEMICOLON:-}" ]; then
      NONAGENTIC_TASKS="${NONAGENTIC_TASKS_SEMICOLON//;/,}"
    else
      NONAGENTIC_TASKS="${NONAGENTIC_TASKS:-MATH500,ifeval,gsm8k,mmlu,hellaswag,HumanEvalPlus,MBPPPlus,GPQADiamond,AIME24}"
    fi
    wants_task() { case ",$NONAGENTIC_TASKS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

    wants_task MATH500       && run_eval_task MATH500       "$MATH_TOKENS" math500       local-chat-completions /v1/chat/completions 0
    wants_task ifeval        && run_eval_task ifeval        2048           ifeval        local-chat-completions /v1/chat/completions 0
    wants_task gsm8k         && run_eval_task gsm8k         512            gsm8k         local-chat-completions /v1/chat/completions 5
    wants_task mmlu          && run_eval_task mmlu          256            mmlu          local-completions      /v1/completions      5
    wants_task hellaswag     && run_eval_task hellaswag     256            hellaswag     local-completions      /v1/completions      10
    wants_task HumanEvalPlus && run_eval_task HumanEvalPlus 1024           humanevalplus local-chat-completions /v1/chat/completions 0 1
    wants_task MBPPPlus      && run_eval_task MBPPPlus      1024           mbppplus      local-chat-completions /v1/chat/completions 0 1
    wants_task GPQADiamond   && run_eval_task GPQADiamond   "$MATH_TOKENS" gpqa_diamond  local-chat-completions /v1/chat/completions 0
    wants_task AIME24        && run_eval_task AIME24        "$MATH_TOKENS" aime24        local-chat-completions /v1/chat/completions 0
  fi
  echo "GRUG_NONAGENTIC_EVAL_OK results=$EVAL_ROOT"
fi

if [ "$RUN_AGENTIC" = "1" ]; then
  # Reproduce the public control-matched evaluation from Marin issue #7743.
  # Serving uses the Marin vLLM fork above; Harbor is pinned independently.
  HARBOR_BIN="${HARBOR_BIN:-/scratch/10635/penfever/miniconda3/envs/otagent/bin/harbor}"
  HARBOR_SRC="${HARBOR_SRC:-$SCRATCH/harbor-issue7743/src}"
  HARBOR_CONFIG="${HARBOR_CONFIG:-$REPO_ROOT/hpc/harbor_yaml/eval/configs/openthoughts_tblite_issue7743.yaml}"
  DATASET_REPO="open-thoughts/OpenThoughts-TBLite"
  DATASET_DIR="${DATASET_DIR:-$SCRATCH/data/datasets/OpenThoughts-TBLite-main}"
  HARBOR_JOBS_DIR="${HARBOR_JOBS_DIR:-$SCRATCH/eval_jobs}"
  HARBOR_CONCURRENCY="${HARBOR_CONCURRENCY:-128}"
  RUN_TAG="issue7743-${SLURM_JOB_ID}-sft-repro-thinking-step630-nemotron-terminal-step1888-openthoughts-tblite"

  test -x "$HARBOR_BIN"
  test -f "$HARBOR_SRC/harbor/__init__.py"
  test -f "$HARBOR_CONFIG"
  source "$SCRATCH/keys.env"
  export DAYTONA_API_KEY="${DAYTONA_DATA_API_KEY:-${DAYTONA_API_KEY:?DAYTONA key is required}}"

  mkdir -p "$DATASET_DIR" "$HARBOR_JOBS_DIR"
  DATASET_LOG="$RUN_DIR/dataset-download.log"
  "$PYTHON" "$REPO_ROOT/eval/snapshot_download.py" "$DATASET_REPO" \
    --local-dir "$DATASET_DIR" >"$DATASET_LOG" 2>&1
  DATASET_PATH=$(grep 'DATASET_PATH=' "$DATASET_LOG" | tail -1 | cut -d= -f2-)
  test -n "$DATASET_PATH"
  TASK_COUNT=$(find "$DATASET_PATH" -mindepth 2 -maxdepth 2 -name instruction.md | wc -l)
  if [ "$TASK_COUNT" -ne 100 ]; then
    echo "FATAL: expected 100 OpenThoughts-TBLite tasks, got $TASK_COUNT" >&2
    exit 2
  fi

  MODEL_INFO='{"max_input_tokens":24576,"max_output_tokens":8192,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
  export PYTHONPATH="$HARBOR_SRC:$REPO_ROOT"
  echo "ISSUE7743_EVAL_START tasks=$TASK_COUNT attempts=3 concurrency=$HARBOR_CONCURRENCY"
  "$HARBOR_BIN" jobs start \
    -p "$DATASET_PATH" \
    --n-concurrent "$HARBOR_CONCURRENCY" \
    --agent terminus-2 \
    --model "hosted_vllm/$MODEL_REPO" \
    --env daytona \
    --agent-kwarg "api_base=http://${HEAD_IP}:${API_PORT}/v1" \
    --agent-kwarg key=fake_key \
    --agent-kwarg temperature=0.6 \
    --agent-kwarg "model_info=$MODEL_INFO" \
    --n-attempts 3 \
    --job-name "$RUN_TAG" \
    --jobs-dir "$HARBOR_JOBS_DIR" \
    --config "$HARBOR_CONFIG"
  echo "ISSUE7743_EVAL_OK run=$HARBOR_JOBS_DIR/$RUN_TAG"
fi
