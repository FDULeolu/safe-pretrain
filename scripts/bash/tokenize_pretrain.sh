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
WORLD_NAME="synthetic_world_1024effects_2048causes_0.1restricted_2arity_strict_wo_overlap_dic-words"
OPEN_FORWARD_WEIGHT=0.30
OPEN_REVERSE_WEIGHT=0.30
OPEN_BIDIRECTIONAL_WEIGHT=0.40
TRAIN_FRACTION=0.99
GENERATOR_VERSION="composition_v1"
PRETRAIN_WRAPPER_VERSION="pretrain_descriptive_v2"
PRETRAIN_CAUSE_ORDER="random_swap"
RENDER_NAME="open_${OPEN_FORWARD_WEIGHT}forward_${OPEN_REVERSE_WEIGHT}reverse_${OPEN_BIDIRECTIONAL_WEIGHT}bi_${TRAIN_FRACTION}train_${GENERATOR_VERSION}_${PRETRAIN_WRAPPER_VERSION}_${PRETRAIN_CAUSE_ORDER}"
DATASET_ROOT="${ROOT_DIR}/data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}"

# Tokenization knobs.
BLOCK_SIZE=512
NUM_PROC=64
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
