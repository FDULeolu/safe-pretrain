#!/usr/bin/env bash
set -euo pipefail

# Evaluate a QA SFT final model on safe test and restricted attack sets.
# Edit the variables below, then run:
#   bash scripts/bash/run_eval_sft_qa.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL="outputs/smollm2-135m-sft-reverse-qa-v1/final_model"
SFT_DIR="data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap/sft/qa_1ex_0.8train_composition_v1"

DEVICE="auto"
DTYPE="auto"
BATCH_SIZE=64
MAX_NEW_TOKENS=32
MAX_EXAMPLES=null
SEED=42

export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

python scripts/python/eval_sft_qa.py \
  --model "${MODEL}" \
  --sft-dir "${SFT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-examples "${MAX_EXAMPLES}" \
  --seed "${SEED}"
