#!/usr/bin/env bash
set -euo pipefail

# Create fixed synthetic world groundtruth.
# Edit the variables below, then run:
#   bash scripts/bash/create_world.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/synthetic_world.yaml"

NUM_EFFECTS=4096
NUM_CAUSES=8192
RECIPE_ARITY=2
RESTRICTED_FRACTION=0.5
SEED=42

WORLD_NAME="synthetic_world_${NUM_EFFECTS}effects_${NUM_CAUSES}causes_${RESTRICTED_FRACTION}restricted_${RECIPE_ARITY}arity_strict_wo_overlap"
OUTPUT_DIR="data/worlds/${WORLD_NAME}"

CAUSE_VOCAB_SOURCE="neutral_words"
EFFECT_VOCAB_SOURCE="generated_phrases"
USE_FAMILIES=false
FORBID_HARMFUL_REAL_TERMS=true
ENFORCE_VOCAB_DISJOINT=true

OVERWRITE=false

ARGS=(
  --config "${CONFIG}"
  "world.name=${WORLD_NAME}"
  "world.seed=${SEED}"
  "world.output_dir=${OUTPUT_DIR}"
  "world.num_effects=${NUM_EFFECTS}"
  "world.num_causes=${NUM_CAUSES}"
  "world.recipe_arity=${RECIPE_ARITY}"
  "partition.restricted_fraction=${RESTRICTED_FRACTION}"
  "surface.cause_vocab_source=${CAUSE_VOCAB_SOURCE}"
  "surface.effect_vocab_source=${EFFECT_VOCAB_SOURCE}"
  "surface.use_families=${USE_FAMILIES}"
  "surface.forbid_harmful_real_terms=${FORBID_HARMFUL_REAL_TERMS}"
  "surface.enforce_vocab_disjoint=${ENFORCE_VOCAB_DISJOINT}"
)

if [[ "${OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

echo "Creating synthetic world:"
echo "  name: ${WORLD_NAME}"
echo "  output: ${OUTPUT_DIR}"
echo "  effects: ${NUM_EFFECTS}"
echo "  causes: ${NUM_CAUSES}"
echo "  arity: ${RECIPE_ARITY}"
echo "  restricted fraction: ${RESTRICTED_FRACTION}"
echo "  seed: ${SEED}"
echo "  overwrite: ${OVERWRITE}"

python scripts/python/create_synthetic_world.py "${ARGS[@]}"
