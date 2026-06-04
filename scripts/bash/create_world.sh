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
RECIPE_ARITY=1
RESTRICTED_FRACTION=0.5
SEED=42

WORLD_NAME="synthetic_world_${NUM_EFFECTS}effects_${NUM_CAUSES}causes_${RESTRICTED_FRACTION}restricted_${RECIPE_ARITY}arity_strict_wo_overlap_dic-words"
OUTPUT_DIR="data/worlds/${WORLD_NAME}"

# Existing pseudo-word setting:
#   CAUSE_VOCAB_SOURCE="neutral_words"
#   EFFECT_VOCAB_SOURCE="generated_phrases"
# Tokenizer-friendly English setting:
#   CAUSE_VOCAB_SOURCE="tokenizer_english_words"
#   EFFECT_VOCAB_SOURCE="tokenizer_english_words"
CAUSE_VOCAB_SOURCE="tokenizer_english_words"
EFFECT_VOCAB_SOURCE="tokenizer_english_words"
TOKENIZER_NAME_OR_PATH="HuggingFaceTB/SmolLM2-135M"
ENGLISH_WORDS_PATH="/usr/share/dict/words"
ENGLISH_MIN_CHARS=6
ENGLISH_MAX_CHARS=12
ENGLISH_SINGLE_TOKEN=true
ENGLISH_RANK_SKIP=960
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
  "surface.tokenizer_name_or_path=${TOKENIZER_NAME_OR_PATH}"
  "surface.english_words_path=${ENGLISH_WORDS_PATH}"
  "surface.english_min_chars=${ENGLISH_MIN_CHARS}"
  "surface.english_max_chars=${ENGLISH_MAX_CHARS}"
  "surface.english_single_token=${ENGLISH_SINGLE_TOKEN}"
  "surface.english_rank_skip=${ENGLISH_RANK_SKIP}"
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
echo "  cause vocab source: ${CAUSE_VOCAB_SOURCE}"
echo "  effect vocab source: ${EFFECT_VOCAB_SOURCE}"
echo "  tokenizer: ${TOKENIZER_NAME_OR_PATH}"
echo "  seed: ${SEED}"
echo "  overwrite: ${OVERWRITE}"

python scripts/python/create_synthetic_world.py "${ARGS[@]}"
