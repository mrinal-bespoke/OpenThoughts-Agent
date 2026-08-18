#!/bin/bash
#SBATCH --account=CCR24067
#SBATCH --partition=gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
# Keep this aligned with Vista's maintained node_exclusion_list in hpc/hpc.py.
# These nodes/racks have produced confirmed NODE_FAILs or repeat NCCL wedges.
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c636-121,c635-101,c641-061,c611-051,c636-152,c608-042,c634-142,c641-012,c634-052,c621-102,c637-042,c638-101,c638-102,c638-112,c639-[001-002,011-012,021-022,031-032,041-042,051-052,061-062,071-072,081-082,091-092,101-102,111-112,121-122,131-132,141-142,151-152]
#SBATCH --time=04:00:00
#SBATCH --job-name=grug-vllm-smoke
#SBATCH --output=/scratch/11694/mkumar73/logs/%x.%j.log

# Two-node Vista smoke test for the repaired Grug/Snowball HF export.
# Grug's supported CUDA layout is TP=1 with experts sharded across DP ranks;
# Vista provides one 96 GB GH200 per node, hence DP=2 + EP across two nodes.

set -euo pipefail

MODEL_REPO="${MODEL_REPO:-laion/grug-67b-a2b-sft-s3-agentic-step1903-repaired}"
MODEL_REVISION="${MODEL_REVISION:-0cb5c3491df5d8ae33ca5b1be513629a06c68882}"
EVAL_ENV="${EVAL_ENV:-$SCRATCH/envs/grug-eval}"
EVALCHEMY_ENV="${EVALCHEMY_ENV:-$SCRATCH/envs/grug-evalchemy}"
REPO_ROOT="${REPO_ROOT:-$SCRATCH/OpenThoughts-Agent}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
RUN_EVAL="${RUN_EVAL:-0}"
RUN_AGENTIC="${RUN_AGENTIC:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
if [ "$RUN_AGENTIC" = "1" ]; then
  RUN_KIND=agentic
elif [ "$RUN_EVAL" = "1" ]; then
  RUN_KIND=eval
else
  RUN_KIND=smoke
fi
RUN_DIR="$SCRATCH/experiments/grug-nonagentic-eval/${RUN_KIND}-${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR" "$SCRATCH/logs" "$SCRATCH/hub"

module purge
module load gcc/14.2.0 cuda/12.8

# The private venv inherits shared Python packages, but venvs do not expose
# console scripts from their base environment. FlashInfer's sampler JIT calls
# `ninja` by name, so keep the shared environment's bin directory on PATH for
# every Ray worker.
export PATH="$EVAL_ENV/bin:/scratch/10635/penfever/miniconda3/envs/otagent/bin:$PATH"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/hub"
export HF_HUB_CACHE="$SCRATCH/hub"
export HF_HUB_OFFLINE=1
export NLTK_DATA="$SCRATCH/nltk_data"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
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
export VLLM_RAY_DP_PACK_STRATEGY=strict
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export VLLM_RINGBUFFER_WARNING_INTERVAL=600
export VLLM_CACHE_ROOT="$RUN_DIR/vllm-cache"
export FLASHINFER_WORKSPACE_BASE="$RUN_DIR/flashinfer-workspace"
export TRITON_CACHE_DIR="$RUN_DIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/inductor-cache"
mkdir -p "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

PYTHON="$EVAL_ENV/bin/python"
test -x "$PYTHON"
test -f "$REPO_ROOT/hpc/vllm/start_vllm_ray_controller.py"

MODEL_PATH=$("$PYTHON" - "$MODEL_REPO" "$MODEL_REVISION" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
PYEOF
)

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_COUNT="${#NODES[@]}"
if [ "$NODE_COUNT" -lt 2 ]; then
  echo "FATAL: expected at least 2 nodes, got $NODE_COUNT" >&2
  exit 2
fi
HEAD_NODE="${NODES[0]}"
HEAD_IP=$(srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$HEAD_NODE" hostname -i | awk '{print $1}')
RAY_ADDRESS="$HEAD_IP:6379"
export RAY_ADDRESS VLLM_HOST_IP="$HEAD_IP"

THINKING_ARGS=()
if [ "$ENABLE_THINKING" = "1" ]; then
  # This checkpoint's shipped chat template uses this kwarg to emit the
  # training-time `Reasoning: /think` control header for chat requests.
  THINKING_ARGS=(--default-chat-template-kwargs '{"enable_thinking": true}')
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
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size "$NODE_COUNT" \
  --data-parallel-address "$HEAD_IP" \
  --data-parallel-size-local 1 \
  --data-parallel-backend ray \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
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

if [ "$RUN_EVAL" = "1" ]; then
  test -x "$EVALCHEMY_ENV/bin/evalchemy"
  EVAL_ROOT="$RUN_DIR/results"
  mkdir -p "$EVAL_ROOT"
  EVAL_MAX_LENGTH=$((MAX_MODEL_LEN - 64))
  MODEL_ARGS="model=${MODEL_REPO},base_url=http://${HEAD_IP}:${API_PORT}/v1/chat/completions,tokenizer=${MODEL_PATH},tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=4,max_retries=8,timeout=1800,max_length=${EVAL_MAX_LENGTH}"

  run_eval_task() {
    local task="$1"
    local max_tokens="$2"
    local output_name="$3"
    echo "EVAL_START task=$task max_tokens=$max_tokens"
    "$EVALCHEMY_ENV/bin/evalchemy" \
      --model local-chat-completions \
      --model_args "$MODEL_ARGS" \
      --tasks "$task" \
      --gen_kwargs "max_gen_toks=${max_tokens}" \
      --max_tokens "$max_tokens" \
      --output_path "$EVAL_ROOT/$output_name" \
      --log_samples \
      --verbosity INFO \
      --num_fewshot 0 \
      --batch_size 1 \
      --seed 1234 \
      --apply_chat_template
    echo "EVAL_DONE task=$task"
  }

  MATH_TOKENS=8192
  if [ "$MATH_TOKENS" -gt "$((MAX_MODEL_LEN - 1024))" ]; then
    MATH_TOKENS=$((MAX_MODEL_LEN - 1024))
  fi
  run_eval_task MATH500 "$MATH_TOKENS" math500
  run_eval_task ifeval 2048 ifeval
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
  MODEL_SLUG="${MODEL_REPO##*/}"
  RUN_TAG="issue7743-${SLURM_JOB_ID}-${MODEL_SLUG}-terminus2-openthoughts-tblite"

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
