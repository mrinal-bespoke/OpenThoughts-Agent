#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=jupiter-smoke-v3a-32k
#SBATCH --output=/e/scratch/reformo/kumar37/jupiter-smoke-v3a-20261027/slurm-%j.log
#SBATCH --export=NONE
#SBATCH --no-requeue
set -euo pipefail
umask 077
ulimit -c 0
BASE=/e/scratch/reformo/kumar37/jupiter-vista-parity-20261018
SMOKE=/e/scratch/reformo/kumar37/jupiter-smoke-v3a-20261027
cd "$SMOKE"
unset PYTHONPATH PYTHONHOME LD_PRELOAD CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_HOME="$BASE/toolchain"
export CC="$BASE/toolchain/bin/aarch64-conda-linux-gnu-gcc"
export CXX="$BASE/toolchain/bin/aarch64-conda-linux-gnu-g++"
export TRITON_CC="$CC"
export PATH="$BASE/serving/bin:$BASE/toolchain/bin:$PATH"
export LD_LIBRARY_PATH="$BASE/serving/lib/python3.12/site-packages/torch/lib"
# Stop only this smoke's proxy after the allocation exits.
cleanup_smoke() {
  touch "$SMOKE/stop-proxy"
  # Only the disposable credential copies created for this smoke are removed.
  rm -f -- "$SMOKE/private.json" "$SMOKE/proxychains.conf"
}
trap cleanup_smoke EXIT
srun --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=18 --gpus-per-task=1 \
  --gpu-bind=single:1 --kill-on-bad-exit=1 \
  "$BASE/harness/bin/python" "$SMOKE/smoke.py" run
