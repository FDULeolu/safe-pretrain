#!/usr/bin/env bash
set -euo pipefail

# Evaluate a checkpoint on synthetic world facts.
# Edit the variables below, then run:
#   bash scripts/run_eval_world.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CHECKPOINT="outputs/smollm2-135m-scratch-0p3b-1epoch-bs512/checkpoints/step-0001209"
PRETRAIN_DIR="data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap/pretrain/0.0reverse_0.99train_4tpl_canonical"

# Set to null for full eval. For a quick check, use e.g. 128.
MAX_EXAMPLES=null
MAX_PER_PARTITION=null

DEVICE="auto"
DTYPE="auto"
MAX_NEW_TOKENS=24
BATCH_SIZE=256
SEED=42

export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

python scripts/eval_world.py \
  --checkpoint "${CHECKPOINT}" \
  --pretrain-dir "${PRETRAIN_DIR}" \
  --max-examples "${MAX_EXAMPLES}" \
  --max-per-partition "${MAX_PER_PARTITION}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}"
