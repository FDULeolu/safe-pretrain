#!/usr/bin/env bash
set -euo pipefail

# Render pretrain JSONL from a fixed synthetic world.
# Edit the variables below, then run:
#   bash scripts/bash/render_pretrain_data.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/synthetic_pretrain_render.yaml"

WORLD_NAME="synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap"

TARGET_TOKENS=300000000
TARGET_RECORDS=null
ESTIMATED_TOKENS_PER_RECORD=24
TRAIN_FRACTION=0.99
VALIDATION_FRACTION=0.01
REVERSE_RATIO=0.0

GENERATOR_VERSION="composition_v1"
CONNECTOR_VERSION="connector_v1"
PRETRAIN_WRAPPER_VERSION="pretrain_descriptive_v1"

OVERWRITE=false

RENDER_NAME="${REVERSE_RATIO}reverse_${TRAIN_FRACTION}train_${GENERATOR_VERSION}"
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
  "pretrain.train_fraction=${TRAIN_FRACTION}"
  "pretrain.validation_fraction=${VALIDATION_FRACTION}"
  "pretrain.reverse_ratio=${REVERSE_RATIO}"
  "composition.generator_version=${GENERATOR_VERSION}"
  "composition.connector_version=${CONNECTOR_VERSION}"
  "composition.pretrain_wrapper_version=${PRETRAIN_WRAPPER_VERSION}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

echo "Rendering pretrain data:"
echo "  world: ${WORLD_NAME}"
echo "  output: ${OUTPUT_DIR}"
echo "  target tokens: ${TARGET_TOKENS}"
echo "  reverse ratio: ${REVERSE_RATIO}"
echo "  connector: ${CONNECTOR_VERSION}"
echo "  wrapper: ${PRETRAIN_WRAPPER_VERSION}"
echo "  overwrite: ${OVERWRITE}"

python scripts/python/render_synthetic_pretrain.py "${ARGS[@]}"
