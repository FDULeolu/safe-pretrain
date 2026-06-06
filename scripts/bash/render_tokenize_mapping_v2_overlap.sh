#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

WORLD_NAME="synthetic_world_1024effects_512causes_0.1restricted_2arity_4x_overlap_dic-words"
RENDER_NAME="mapping_v2_0p3b_1024rel_512cause_4x_overlap_composable_v1_v3_random_swap"
DATASET_ROOT="${ROOT_DIR}/data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}"
TOKENIZED_PATH="${DATASET_ROOT}/tokenized/bs512"
LOG_DIR="${ROOT_DIR}/logs/pretrain_data"
LOG_FILE="${LOG_DIR}/${RENDER_NAME}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "[$(date -Is)] Starting mapping-v2 overlap render/tokenize"
echo "root: ${ROOT_DIR}"
echo "world: ${WORLD_NAME}"
echo "render: ${RENDER_NAME}"
echo "dataset_root: ${DATASET_ROOT}"
echo "tokenized_path: ${TOKENIZED_PATH}"
echo "log_file: ${LOG_FILE}"

echo "[$(date -Is)] Rendering pretrain data"
conda run --no-capture-output -n safe-pretrain \
  python scripts/python/render_synthetic_pretrain.py \
    --config configs/synthetic_pretrain_render.yaml \
    "world.name=${WORLD_NAME}" \
    "world.path=data/worlds/${WORLD_NAME}" \
    "render.name=${RENDER_NAME}" \
    "render.output_dir=data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}" \
    "pretrain.pattern_preset=mapping_v2" \
    "composition.connector_version=connector_composable_v1" \
    "composition.pretrain_wrapper_version=pretrain_descriptive_v3" \
    "composition.pretrain_cause_order=random_swap" \
    "composition.pretrain_alias_enabled=false"

echo "[$(date -Is)] Tokenizing pretrain data"
conda run --no-capture-output -n safe-pretrain \
  python scripts/python/tokenize_dataset.py \
    --config configs/pretrain_a6000_smollm2_135m.yaml \
    "data.raw.dataset_root=${DATASET_ROOT}" \
    "data.tokenized.block_size=512" \
    "data.tokenized.num_proc=64" \
    "data.tokenized.overwrite=false" \
    "data.tokenized.append_eos=true"

echo "[$(date -Is)] Finished mapping-v2 overlap render/tokenize"
