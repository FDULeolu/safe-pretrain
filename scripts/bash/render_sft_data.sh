#!/usr/bin/env bash
set -euo pipefail

# Render chat-format SFT QA data from a fixed synthetic world.
# Edit the variables below, then run:
#   bash scripts/bash/render_sft_data.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/synthetic_sft_qa.yaml"

WORLD_NAME="synthetic_world_4096effects_8192causes_0.5restricted_2arity_strict_wo_overlap"

EXAMPLES_PER_RELATION_PER_TASK=1
TRAIN_FRACTION=0.8
VALIDATION_FRACTION=0.1
RESTRICTED_FORWARD_TRAIN_FRACTION=0.5

GENERATOR_VERSION="composition_v1"
CONNECTOR_VERSION="connector_v1"
SFT_WRAPPER_VERSION="sft_chat_qa_v1"
CHAT_TEMPLATE_ID="smollm2_chatml_v1"

OVERWRITE=false

SFT_NAME="qa_${EXAMPLES_PER_RELATION_PER_TASK}ex_${TRAIN_FRACTION}train_${VALIDATION_FRACTION}val_${GENERATOR_VERSION}_${RESTRICTED_FORWARD_TRAIN_FRACTION}restrict-train"
OUTPUT_DIR="data/worlds/${WORLD_NAME}/sft/${SFT_NAME}"

ARGS=(
  --config "${CONFIG}"
  "world.name=${WORLD_NAME}"
  "world.path=data/worlds/${WORLD_NAME}"
  "sft.name=${SFT_NAME}"
  "sft.output_dir=${OUTPUT_DIR}"
  "sft_data.examples_per_relation_per_task=${EXAMPLES_PER_RELATION_PER_TASK}"
  "sft_data.train_fraction=${TRAIN_FRACTION}"
  "sft_data.validation_fraction=${VALIDATION_FRACTION}"
  "sft_data.restricted_forward_train_fraction=${RESTRICTED_FORWARD_TRAIN_FRACTION}"
  "composition.generator_version=${GENERATOR_VERSION}"
  "composition.connector_version=${CONNECTOR_VERSION}"
  "composition.sft_wrapper_version=${SFT_WRAPPER_VERSION}"
  "composition.chat_template_id=${CHAT_TEMPLATE_ID}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

echo "Rendering SFT QA data:"
echo "  world: ${WORLD_NAME}"
echo "  output: ${OUTPUT_DIR}"
echo "  examples/relation/task: ${EXAMPLES_PER_RELATION_PER_TASK}"
echo "  train fraction: ${TRAIN_FRACTION}"
echo "  validation fraction: ${VALIDATION_FRACTION}"
echo "  restricted forward train fraction: ${RESTRICTED_FORWARD_TRAIN_FRACTION}"
echo "  connector: ${CONNECTOR_VERSION}"
echo "  wrapper: ${SFT_WRAPPER_VERSION}"
echo "  overwrite: ${OVERWRITE}"

python scripts/python/render_synthetic_sft_qa.py "${ARGS[@]}"
