#!/usr/bin/env bash
set -euo pipefail

# Render pretrain JSONL from a fixed synthetic world.
# Edit the variables below, then run:
#   bash scripts/bash/render_pretrain_data.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/synthetic_pretrain_render.yaml"

WORLD_NAME="synthetic_world_1024effects_2048causes_0.1restricted_2arity_strict_wo_overlap_dic-words"

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
# Set to custom to use the explicit weights below. Other options:
# vanilla, bidirectional, mapping_v1, mapping_v2, mirror_probe_v1, mirror_probe_v2.
PATTERN_PRESET="custom"
# Open-relation pattern weights. Restricted relations keep direct reverse closed.
OPEN_FORWARD_WEIGHT=0.20
OPEN_REVERSE_WEIGHT=0.20
OPEN_IDENTITY_WEIGHT=0.20
OPEN_FORWARD_REVERSE_WEIGHT=0.20
OPEN_REVERSE_FORWARD_WEIGHT=0.20
OPEN_BIDIRECTIONAL_WEIGHT=0.0
RESTRICTED_FORWARD_WEIGHT=0.25
RESTRICTED_IDENTITY_WEIGHT=0.25
RESTRICTED_FORWARD_REVERSE_WEIGHT=0.25
RESTRICTED_REVERSE_FORWARD_WEIGHT=0.25
REVERSE_RATIO=0.0

GENERATOR_VERSION="composition_v1"
# Use connector_composable_v1 with PATTERN_PRESET=mapping_v2.
CONNECTOR_VERSION="connector_v1"
PRETRAIN_WRAPPER_VERSION="pretrain_descriptive_v2"
PRETRAIN_CAUSE_ORDER="random_swap"
PRETRAIN_ALIAS_ENABLED=false
PRETRAIN_ALIAS_REPLACEMENT_PROBABILITY=0.25
PRETRAIN_ANSWER_ALIAS_REPLACEMENT_PROBABILITY=0.0
BIDIRECTIONAL_FORWARD_FIRST_WEIGHT=0.5
BIDIRECTIONAL_REVERSE_FIRST_WEIGHT=0.5

OVERWRITE=false
EXPORT_RAW_ARROW=false
RAW_ARROW_TEXT_ONLY=true

if [[ "${PRETRAIN_ALIAS_ENABLED}" == "true" ]]; then
  ALIAS_SUFFIX="aliasp${PRETRAIN_ALIAS_REPLACEMENT_PROBABILITY}"
else
  ALIAS_SUFFIX="noalias"
fi
RENDER_NAME="${PATTERN_PRESET}_bridge_open_${OPEN_FORWARD_WEIGHT}f_${OPEN_REVERSE_WEIGHT}r_${OPEN_IDENTITY_WEIGHT}i_${OPEN_FORWARD_REVERSE_WEIGHT}fr_${OPEN_REVERSE_FORWARD_WEIGHT}rf_restrict_${RESTRICTED_FORWARD_WEIGHT}f_${RESTRICTED_IDENTITY_WEIGHT}i_${RESTRICTED_FORWARD_REVERSE_WEIGHT}fr_${RESTRICTED_REVERSE_FORWARD_WEIGHT}rf_${ALIAS_SUFFIX}_${TRAIN_FRACTION}train_${GENERATOR_VERSION}_${CONNECTOR_VERSION}_${PRETRAIN_WRAPPER_VERSION}_${PRETRAIN_CAUSE_ORDER}"
OUTPUT_DIR="data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}"

ARGS=(
  --config "${CONFIG}"
  "world.name=${WORLD_NAME}"
  "world.path=data/worlds/${WORLD_NAME}"
  "render.name=${RENDER_NAME}"
  "render.output_dir=${OUTPUT_DIR}"
  "render.export_raw_arrow=${EXPORT_RAW_ARROW}"
  "render.raw_arrow_dir=${OUTPUT_DIR}/raw_arrow"
  "render.raw_arrow_text_only=${RAW_ARROW_TEXT_ONLY}"
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
  "pretrain.pattern_preset=${PATTERN_PRESET}"
  "pretrain.open_pattern_weights.forward=${OPEN_FORWARD_WEIGHT}"
  "pretrain.open_pattern_weights.reverse=${OPEN_REVERSE_WEIGHT}"
  "pretrain.open_pattern_weights.identity=${OPEN_IDENTITY_WEIGHT}"
  "pretrain.open_pattern_weights.forward_reverse=${OPEN_FORWARD_REVERSE_WEIGHT}"
  "pretrain.open_pattern_weights.reverse_forward=${OPEN_REVERSE_FORWARD_WEIGHT}"
  "pretrain.open_pattern_weights.bidirectional=${OPEN_BIDIRECTIONAL_WEIGHT}"
  "pretrain.restricted_pattern_weights.forward=${RESTRICTED_FORWARD_WEIGHT}"
  "pretrain.restricted_pattern_weights.identity=${RESTRICTED_IDENTITY_WEIGHT}"
  "pretrain.restricted_pattern_weights.forward_reverse=${RESTRICTED_FORWARD_REVERSE_WEIGHT}"
  "pretrain.restricted_pattern_weights.reverse_forward=${RESTRICTED_REVERSE_FORWARD_WEIGHT}"
  "pretrain.restricted_pattern_weights.reverse=0.0"
  "pretrain.restricted_pattern_weights.bidirectional=0.0"
  "pretrain.reverse_ratio=${REVERSE_RATIO}"
  "composition.generator_version=${GENERATOR_VERSION}"
  "composition.connector_version=${CONNECTOR_VERSION}"
  "composition.pretrain_wrapper_version=${PRETRAIN_WRAPPER_VERSION}"
  "composition.pretrain_cause_order=${PRETRAIN_CAUSE_ORDER}"
  "composition.pretrain_alias_enabled=${PRETRAIN_ALIAS_ENABLED}"
  "composition.pretrain_alias_replacement_probability=${PRETRAIN_ALIAS_REPLACEMENT_PROBABILITY}"
  "composition.pretrain_answer_alias_replacement_probability=${PRETRAIN_ANSWER_ALIAS_REPLACEMENT_PROBABILITY}"
  "composition.bidirectional_order_weights.forward_first=${BIDIRECTIONAL_FORWARD_FIRST_WEIGHT}"
  "composition.bidirectional_order_weights.reverse_first=${BIDIRECTIONAL_REVERSE_FIRST_WEIGHT}"
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
echo "  pattern preset: ${PATTERN_PRESET}"
echo "  open pattern weights: forward=${OPEN_FORWARD_WEIGHT}, reverse=${OPEN_REVERSE_WEIGHT}, identity=${OPEN_IDENTITY_WEIGHT}, forward_reverse=${OPEN_FORWARD_REVERSE_WEIGHT}, reverse_forward=${OPEN_REVERSE_FORWARD_WEIGHT}, bidirectional=${OPEN_BIDIRECTIONAL_WEIGHT}"
echo "  restricted pattern weights: forward=${RESTRICTED_FORWARD_WEIGHT}, identity=${RESTRICTED_IDENTITY_WEIGHT}, forward_reverse=${RESTRICTED_FORWARD_REVERSE_WEIGHT}, reverse_forward=${RESTRICTED_REVERSE_FORWARD_WEIGHT}"
echo "  legacy reverse ratio fallback: ${REVERSE_RATIO}"
echo "  connector: ${CONNECTOR_VERSION}"
echo "  wrapper: ${PRETRAIN_WRAPPER_VERSION}"
echo "  cause order: ${PRETRAIN_CAUSE_ORDER}"
echo "  alias: enabled=${PRETRAIN_ALIAS_ENABLED}, replacement_probability=${PRETRAIN_ALIAS_REPLACEMENT_PROBABILITY}, answer_probability=${PRETRAIN_ANSWER_ALIAS_REPLACEMENT_PROBABILITY}"
echo "  bidirectional order weights: forward_first=${BIDIRECTIONAL_FORWARD_FIRST_WEIGHT}, reverse_first=${BIDIRECTIONAL_REVERSE_FIRST_WEIGHT}"
echo "  overwrite: ${OVERWRITE}"
echo "  export raw arrow: ${EXPORT_RAW_ARROW}"

python scripts/python/render_synthetic_pretrain.py "${ARGS[@]}"
