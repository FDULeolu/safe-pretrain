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
OPEN_FORWARD_WEIGHT=0.20
OPEN_REVERSE_WEIGHT=0.20
OPEN_IDENTITY_WEIGHT=0.20
OPEN_FORWARD_REVERSE_WEIGHT=0.20
OPEN_REVERSE_FORWARD_WEIGHT=0.20
RESTRICTED_FORWARD_WEIGHT=0.25
RESTRICTED_IDENTITY_WEIGHT=0.25
RESTRICTED_FORWARD_REVERSE_WEIGHT=0.25
RESTRICTED_REVERSE_FORWARD_WEIGHT=0.25
TRAIN_FRACTION=0.99
GENERATOR_VERSION="composition_v1"
PRETRAIN_WRAPPER_VERSION="pretrain_descriptive_v2"
PRETRAIN_CAUSE_ORDER="random_swap"
PRETRAIN_ALIAS_ENABLED=false
PRETRAIN_ALIAS_REPLACEMENT_PROBABILITY=0.25
if [[ "${PRETRAIN_ALIAS_ENABLED}" == "true" ]]; then
  ALIAS_SUFFIX="aliasp${PRETRAIN_ALIAS_REPLACEMENT_PROBABILITY}"
else
  ALIAS_SUFFIX="noalias"
fi
RENDER_NAME="bridge_open_${OPEN_FORWARD_WEIGHT}f_${OPEN_REVERSE_WEIGHT}r_${OPEN_IDENTITY_WEIGHT}i_${OPEN_FORWARD_REVERSE_WEIGHT}fr_${OPEN_REVERSE_FORWARD_WEIGHT}rf_restrict_${RESTRICTED_FORWARD_WEIGHT}f_${RESTRICTED_IDENTITY_WEIGHT}i_${RESTRICTED_FORWARD_REVERSE_WEIGHT}fr_${RESTRICTED_REVERSE_FORWARD_WEIGHT}rf_${ALIAS_SUFFIX}_${TRAIN_FRACTION}train_${GENERATOR_VERSION}_${PRETRAIN_WRAPPER_VERSION}_${PRETRAIN_CAUSE_ORDER}"
DATASET_ROOT="${ROOT_DIR}/data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}"

# Tokenization knobs.
BLOCK_SIZE=512
NUM_PROC=64
OVERWRITE=false
APPEND_EOS=true
USE_RAW_ARROW=false

# Keep Hugging Face cache local to this repo.
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

TOKENIZED_PATH="${DATASET_ROOT}/tokenized/bs${BLOCK_SIZE}"
RAW_FORMAT="jsonl"
RAW_DATASET_ROOT="${DATASET_ROOT}"
if [[ "${USE_RAW_ARROW}" == "true" ]]; then
  RAW_FORMAT="hf_disk"
  RAW_DATASET_ROOT="${DATASET_ROOT}/raw_arrow"
fi

if [[ "${USE_RAW_ARROW}" == "true" ]]; then
  if [[ ! -d "${RAW_DATASET_ROOT}" ]]; then
    echo "Missing raw Arrow dataset: ${RAW_DATASET_ROOT}" >&2
    exit 1
  fi
elif [[ ! -f "${DATASET_ROOT}/pretrain_train.jsonl" ]]; then
  echo "Missing train file: ${DATASET_ROOT}/pretrain_train.jsonl" >&2
  exit 1
elif [[ ! -f "${DATASET_ROOT}/pretrain_validation.jsonl" ]]; then
  echo "Missing validation file: ${DATASET_ROOT}/pretrain_validation.jsonl" >&2
  exit 1
fi

echo "Tokenizing pretrain dataset:"
echo "  dataset root: ${DATASET_ROOT}"
echo "  raw format: ${RAW_FORMAT}"
echo "  raw dataset root: ${RAW_DATASET_ROOT}"
echo "  tokenized path: ${TOKENIZED_PATH}"
echo "  block size: ${BLOCK_SIZE}"
echo "  num proc: ${NUM_PROC}"
echo "  overwrite: ${OVERWRITE}"

python scripts/python/tokenize_dataset.py \
  --config "${CONFIG}" \
  "data.raw.format=${RAW_FORMAT}" \
  "data.raw.dataset_root=${RAW_DATASET_ROOT}" \
  "data.tokenized.block_size=${BLOCK_SIZE}" \
  "data.tokenized.path=${TOKENIZED_PATH}" \
  "data.tokenized.num_proc=${NUM_PROC}" \
  "data.tokenized.overwrite=${OVERWRITE}" \
  "data.tokenized.append_eos=${APPEND_EOS}"
