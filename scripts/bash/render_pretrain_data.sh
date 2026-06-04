#!/usr/bin/env bash
set -euo pipefail

# Render pretrain JSONL from a fixed synthetic world.
# Edit the variables below, then run:
#   bash scripts/bash/render_pretrain_data.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/synthetic_pretrain_render.yaml"

WORLD_NAME="synthetic_world_4096effects_8192causes_0.5restricted_2arity_strict_wo_overlap"

TARGET_TOKENS=300000000
TARGET_RECORDS=null
ESTIMATED_TOKENS_PER_RECORD=24
TOKEN_BUDGETING_ENABLED=true
TOKEN_BUDGETING_STRATEGY="estimate_records"
TOKENIZER_NAME_OR_PATH="HuggingFaceTB/SmolLM2-135M"
TOKEN_BUDGET_TARGET_SPLIT="train"
TOKEN_BUDGET_APPEND_EOS=true
TOKEN_BUDGET_COUNT_BATCH_SIZE=4096
TOKEN_BUDGET_ESTIMATE_SAMPLE_RECORDS=200000
TOKEN_BUDGET_RECORD_SAFETY_MARGIN=1.001
TRAIN_FRACTION=0.99
VALIDATION_FRACTION=0.01
REVERSE_RATIO=0.25

GENERATOR_VERSION="composition_v1"
CONNECTOR_VERSION="connector_v1"
PRETRAIN_WRAPPER_VERSION="pretrain_descriptive_v2"
PRETRAIN_CAUSE_ORDER="random_swap"

OVERWRITE=false

RENDER_NAME="${REVERSE_RATIO}reverse_${TRAIN_FRACTION}train_${GENERATOR_VERSION}_${PRETRAIN_WRAPPER_VERSION}_${PRETRAIN_CAUSE_ORDER}"
OUTPUT_DIR="data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}"

ARGS=(
  --config "${CONFIG}"
  "world.name=${WORLD_NAME}"
  "world.path=data/worlds/${WORLD_NAME}"
  "render.name=${RENDER_NAME}"
  "render.output_dir=${OUTPUT_DIR}"
  "pretrain.target_tokens=${TARGET_TOKENS}"
  "pretrain.target_records=${TARGET_RECORDS}"
  "pretrain.estimated_tokens_per_record=${ESTIMATED_TOKENS_PER_RECORD}"
  "pretrain.token_budgeting.enabled=${TOKEN_BUDGETING_ENABLED}"
  "pretrain.token_budgeting.strategy=${TOKEN_BUDGETING_STRATEGY}"
  "pretrain.token_budgeting.tokenizer_name_or_path=${TOKENIZER_NAME_OR_PATH}"
  "pretrain.token_budgeting.target_split=${TOKEN_BUDGET_TARGET_SPLIT}"
  "pretrain.token_budgeting.append_eos=${TOKEN_BUDGET_APPEND_EOS}"
  "pretrain.token_budgeting.count_batch_size=${TOKEN_BUDGET_COUNT_BATCH_SIZE}"
  "pretrain.token_budgeting.estimate_sample_records=${TOKEN_BUDGET_ESTIMATE_SAMPLE_RECORDS}"
  "pretrain.token_budgeting.record_safety_margin=${TOKEN_BUDGET_RECORD_SAFETY_MARGIN}"
  "pretrain.train_fraction=${TRAIN_FRACTION}"
  "pretrain.validation_fraction=${VALIDATION_FRACTION}"
  "pretrain.reverse_ratio=${REVERSE_RATIO}"
  "composition.generator_version=${GENERATOR_VERSION}"
  "composition.connector_version=${CONNECTOR_VERSION}"
  "composition.pretrain_wrapper_version=${PRETRAIN_WRAPPER_VERSION}"
  "composition.pretrain_cause_order=${PRETRAIN_CAUSE_ORDER}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

echo "Rendering pretrain data:"
echo "  world: ${WORLD_NAME}"
echo "  output: ${OUTPUT_DIR}"
echo "  target tokens: ${TARGET_TOKENS}"
echo "  token budget strategy: ${TOKEN_BUDGETING_STRATEGY}"
echo "  token budget target split: ${TOKEN_BUDGET_TARGET_SPLIT}"
echo "  token budget tokenizer: ${TOKENIZER_NAME_OR_PATH}"
echo "  reverse ratio: ${REVERSE_RATIO}"
echo "  connector: ${CONNECTOR_VERSION}"
echo "  wrapper: ${PRETRAIN_WRAPPER_VERSION}"
echo "  cause order: ${PRETRAIN_CAUSE_ORDER}"
echo "  overwrite: ${OVERWRITE}"

python scripts/python/render_synthetic_pretrain.py "${ARGS[@]}"
