#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data3/yizhou/miniconda3/envs/safe-pretrain/bin/python}"
REQUIRE_GPU="${REQUIRE_GPU:-true}"
FINAL_EXPERIMENT_CONCURRENCY="${FINAL_EXPERIMENT_CONCURRENCY:-2}"
MIN_GPUS="${MIN_GPUS:-}"
MIN_FREE_GB="${MIN_FREE_GB:-500}"
RUN_TESTS="${RUN_TESTS:-false}"
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}"

fail() {
  echo "preflight failed: $*" >&2
  exit 2
}

log() {
  echo "[preflight] $*"
}

[[ -x "${PYTHON_BIN}" ]] || fail "PYTHON_BIN is not executable: ${PYTHON_BIN}"

case "${FINAL_EXPERIMENT_CONCURRENCY}" in
  1 | 2) ;;
  *) fail "FINAL_EXPERIMENT_CONCURRENCY must be 1 or 2, got ${FINAL_EXPERIMENT_CONCURRENCY}" ;;
esac
MIN_GPUS="${MIN_GPUS:-$((FINAL_EXPERIMENT_CONCURRENCY * 4))}"

log "checking Python environment and local model cache"
"${PYTHON_BIN}" - <<PY
from safe_pretrain.config import load_config
from transformers import AutoConfig, AutoTokenizer

for path in (
    "configs/synthetic_dataset.yaml",
    "configs/pretrain_a6000_smollm2_135m.yaml",
    "configs/sft_qa_smollm2_135m.yaml",
):
    load_config(path)

AutoTokenizer.from_pretrained("${MODEL_NAME}", local_files_only=True)
AutoConfig.from_pretrained("${MODEL_NAME}", local_files_only=True)
print("python/config/model-cache ok")
PY

log "checking bash syntax"
bash -n scripts/bash/run_experiment_pipeline.sh
bash -n scripts/bash/run_all_final_experiments.sh
for script in $(find scripts/bash/final_experiments -type f -name '*.sh' | sort); do
  bash -n "${script}"
done

log "checking final experiment no-op dry runs"
for script in $(find scripts/bash/final_experiments -type f -name '*.sh' | sort); do
  RUN_DATA=false \
  RUN_PRETRAIN=false \
  RUN_SFT=false \
  RUN_EVAL_PRETRAIN=false \
  RUN_EVAL_SFT=false \
    bash "${script}" >/tmp/safe_pretrain_final_experiment_dry_run.out
done
RUN_DATA=false \
RUN_PRETRAIN=false \
RUN_SFT=false \
RUN_EVAL_PRETRAIN=false \
RUN_EVAL_SFT=false \
CHECK_PORTS=false \
FINAL_EXPERIMENT_CONCURRENCY="${FINAL_EXPERIMENT_CONCURRENCY}" \
LOG_ROOT=/tmp/safe_pretrain_final_experiment_scheduler_dry_run \
  bash scripts/bash/run_all_final_experiments.sh >/tmp/safe_pretrain_final_experiment_scheduler_dry_run.out

log "checking whitespace"
git diff --check

free_gb="$(df -Pk . | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
if (( free_gb < MIN_FREE_GB )); then
  fail "only ${free_gb} GiB free under ${ROOT_DIR}; require at least ${MIN_FREE_GB} GiB"
fi
log "disk ok: ${free_gb} GiB free under ${ROOT_DIR}"

if [[ "${REQUIRE_GPU}" == "true" ]]; then
  log "checking GPU visibility"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi is not available"
  fi
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  if (( gpu_count < MIN_GPUS )); then
    fail "only ${gpu_count} GPU(s) visible; require at least ${MIN_GPUS}"
  fi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
else
  log "skipping GPU check because REQUIRE_GPU=false"
fi

if [[ "${RUN_TESTS}" == "true" ]]; then
  log "running tests"
  "${PYTHON_BIN}" -m pytest -q
else
  log "skipping tests because RUN_TESTS=false"
fi

log "ready"
