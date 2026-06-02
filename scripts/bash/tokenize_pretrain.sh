#!/usr/bin/env bash
set -euo pipefail

# Tokenize one rendered pretrain dataset.
# Edit the variables below, then run:
#   bash scripts/bash/tokenize_pretrain.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/pretrain_a6000_smollm2_135m.yaml"

# Rendered pretrain dataset directory. It should contain:
#   pretrain_train.jsonl
#   pretrain_validation.jsonl
DATASET_ROOT="/data3/yizhou/projects/safe-pretrain/data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_2arity_strict_wo_overlap/pretrain/0.25reverse_0.99train_composition_v1"

# Tokenization knobs.
BLOCK_SIZE=512
NUM_PROC=32
OVERWRITE=false
APPEND_EOS=true

# Keep Hugging Face cache local to this repo.
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

TOKENIZED_PATH="${DATASET_ROOT}/tokenized/bs${BLOCK_SIZE}"

if [[ ! -f "${DATASET_ROOT}/pretrain_train.jsonl" ]]; then
  echo "Missing train file: ${DATASET_ROOT}/pretrain_train.jsonl" >&2
  exit 1
fi

if [[ ! -f "${DATASET_ROOT}/pretrain_validation.jsonl" ]]; then
  echo "Missing validation file: ${DATASET_ROOT}/pretrain_validation.jsonl" >&2
  exit 1
fi

echo "Tokenizing pretrain dataset:"
echo "  dataset root: ${DATASET_ROOT}"
echo "  tokenized path: ${TOKENIZED_PATH}"
echo "  block size: ${BLOCK_SIZE}"
echo "  num proc: ${NUM_PROC}"
echo "  overwrite: ${OVERWRITE}"

python scripts/python/tokenize_dataset.py \
  --config "${CONFIG}" \
  "data.raw.dataset_root=${DATASET_ROOT}" \
  "data.tokenized.block_size=${BLOCK_SIZE}" \
  "data.tokenized.num_proc=${NUM_PROC}" \
  "data.tokenized.overwrite=${OVERWRITE}" \
  "data.tokenized.append_eos=${APPEND_EOS}"
