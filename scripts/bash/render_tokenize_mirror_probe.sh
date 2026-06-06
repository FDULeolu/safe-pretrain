#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

WORLD_NAME="${WORLD_NAME:-synthetic_world_1024effects_512causes_0.1restricted_2arity_4x_overlap_dic-words}"
PATTERN_PRESET="${PATTERN_PRESET:-mirror_probe_v2}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
NUM_PROC="${NUM_PROC:-64}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-8192}"
TOKENIZE_MODE="${TOKENIZE_MODE:-stream_jsonl}"
EXPORT_RAW_ARROW="${EXPORT_RAW_ARROW:-false}"
SKIP_RENDER="${SKIP_RENDER:-false}"
OVERWRITE_RENDER="${OVERWRITE_RENDER:-false}"
OVERWRITE_TOKENIZED="${OVERWRITE_TOKENIZED:-false}"
TARGET_TOKENS="${TARGET_TOKENS:-300000000}"
TOKEN_BUDGETING_ENABLED="${TOKEN_BUDGETING_ENABLED:-true}"
TOKEN_BUDGETING_STRATEGY="${TOKEN_BUDGETING_STRATEGY:-estimate_records}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-HuggingFaceTB/SmolLM2-135M}"
TOKEN_BUDGET_TARGET_SPLIT="${TOKEN_BUDGET_TARGET_SPLIT:-train}"
TOKEN_BUDGET_APPEND_EOS="${TOKEN_BUDGET_APPEND_EOS:-true}"
TOKEN_BUDGET_COUNT_BATCH_SIZE="${TOKEN_BUDGET_COUNT_BATCH_SIZE:-4096}"
TOKEN_BUDGET_ESTIMATE_SAMPLE_RECORDS="${TOKEN_BUDGET_ESTIMATE_SAMPLE_RECORDS:-200000}"
TOKEN_BUDGET_RECORD_SAFETY_MARGIN="${TOKEN_BUDGET_RECORD_SAFETY_MARGIN:-1.001}"

RENDER_NAME="${RENDER_NAME:-${PATTERN_PRESET}_0p3b_1024rel_512cause_4x_overlap_composable_v1_v3_random_swap_stream}"
DATASET_ROOT="${ROOT_DIR}/data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}"
TOKENIZED_PATH="${DATASET_ROOT}/tokenized/bs${BLOCK_SIZE}"
RAW_ARROW_PATH="${DATASET_ROOT}/raw_arrow"
LOG_DIR="${ROOT_DIR}/logs/pretrain_data"
LOG_FILE="${LOG_DIR}/${RENDER_NAME}.log"

mkdir -p "${LOG_DIR}" "${ROOT_DIR}/.codex-tmp"
exec > >(tee -a "${LOG_FILE}") 2>&1

export TMPDIR="${ROOT_DIR}/.codex-tmp"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
if [[ "${TOKENIZE_MODE}" == "stream_jsonl" ]]; then
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
else
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
fi
export OMP_NUM_THREADS=1

echo "[$(date -Is)] Starting mirror-probe render/tokenize"
echo "root: ${ROOT_DIR}"
echo "world: ${WORLD_NAME}"
echo "preset: ${PATTERN_PRESET}"
echo "render: ${RENDER_NAME}"
echo "dataset_root: ${DATASET_ROOT}"
echo "raw_arrow_path: ${RAW_ARROW_PATH}"
echo "tokenized_path: ${TOKENIZED_PATH}"
echo "target_tokens: ${TARGET_TOKENS}"
echo "tokenize_mode: ${TOKENIZE_MODE}"
echo "tokenize_batch_size: ${TOKENIZE_BATCH_SIZE}"
echo "export_raw_arrow: ${EXPORT_RAW_ARROW}"
echo "log_file: ${LOG_FILE}"

RENDER_ARGS=(
  --config configs/synthetic_pretrain_render.yaml
  "world.name=${WORLD_NAME}"
  "world.path=data/worlds/${WORLD_NAME}"
  "render.name=${RENDER_NAME}"
  "render.output_dir=data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}"
  "render.export_raw_arrow=${EXPORT_RAW_ARROW}"
  "render.raw_arrow_dir=data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}/raw_arrow"
  "render.raw_arrow_text_only=true"
  "pretrain.target_tokens=${TARGET_TOKENS}"
  "pretrain.token_budgeting.enabled=${TOKEN_BUDGETING_ENABLED}"
  "pretrain.token_budgeting.strategy=${TOKEN_BUDGETING_STRATEGY}"
  "pretrain.token_budgeting.tokenizer_name_or_path=${TOKENIZER_NAME_OR_PATH}"
  "pretrain.token_budgeting.target_split=${TOKEN_BUDGET_TARGET_SPLIT}"
  "pretrain.token_budgeting.append_eos=${TOKEN_BUDGET_APPEND_EOS}"
  "pretrain.token_budgeting.count_batch_size=${TOKEN_BUDGET_COUNT_BATCH_SIZE}"
  "pretrain.token_budgeting.estimate_sample_records=${TOKEN_BUDGET_ESTIMATE_SAMPLE_RECORDS}"
  "pretrain.token_budgeting.record_safety_margin=${TOKEN_BUDGET_RECORD_SAFETY_MARGIN}"
  "pretrain.pattern_preset=${PATTERN_PRESET}"
  "composition.connector_version=connector_composable_v1"
  "composition.pretrain_wrapper_version=pretrain_descriptive_v3"
  "composition.pretrain_cause_order=random_swap"
  "composition.pretrain_alias_enabled=false"
)
if [[ "${OVERWRITE_RENDER}" == "true" ]]; then
  RENDER_ARGS+=(--overwrite)
fi

if [[ "${SKIP_RENDER}" == "true" ]]; then
  echo "[$(date -Is)] Skipping render"
else
  echo "[$(date -Is)] Rendering pretrain data"
  conda run --no-capture-output -n safe-pretrain \
    python scripts/python/render_synthetic_pretrain.py "${RENDER_ARGS[@]}"
fi

if [[ "${TOKENIZE_MODE}" == "stream_jsonl" ]]; then
  echo "[$(date -Is)] Stream-tokenizing pretrain JSONL"
  conda run --no-capture-output -n safe-pretrain \
    python scripts/python/tokenize_jsonl_stream.py \
      --config configs/pretrain_a6000_smollm2_135m.yaml \
      --batch-size "${TOKENIZE_BATCH_SIZE}" \
      "data.raw.dataset_root=${DATASET_ROOT}" \
      "data.tokenized.path=${TOKENIZED_PATH}" \
      "data.tokenized.block_size=${BLOCK_SIZE}" \
      "data.tokenized.overwrite=${OVERWRITE_TOKENIZED}" \
      "data.tokenized.append_eos=true"
elif [[ "${TOKENIZE_MODE}" == "raw_arrow" ]]; then
  echo "[$(date -Is)] Tokenizing pretrain data from raw Arrow"
  conda run --no-capture-output -n safe-pretrain \
    python scripts/python/tokenize_dataset.py \
      --config configs/pretrain_a6000_smollm2_135m.yaml \
      "data.raw.format=hf_disk" \
      "data.raw.dataset_root=${RAW_ARROW_PATH}" \
      "data.raw.text_column=text" \
      "data.tokenized.path=${TOKENIZED_PATH}" \
      "data.tokenized.block_size=${BLOCK_SIZE}" \
      "data.tokenized.num_proc=${NUM_PROC}" \
      "data.tokenized.overwrite=${OVERWRITE_TOKENIZED}" \
      "data.tokenized.append_eos=true"
elif [[ "${TOKENIZE_MODE}" == "hf_jsonl" ]]; then
  echo "[$(date -Is)] Tokenizing pretrain JSONL through Hugging Face map"
  conda run --no-capture-output -n safe-pretrain \
    python scripts/python/tokenize_dataset.py \
      --config configs/pretrain_a6000_smollm2_135m.yaml \
      "data.raw.format=jsonl" \
      "data.raw.dataset_root=${DATASET_ROOT}" \
      "data.raw.text_column=text" \
      "data.tokenized.path=${TOKENIZED_PATH}" \
      "data.tokenized.block_size=${BLOCK_SIZE}" \
      "data.tokenized.num_proc=${NUM_PROC}" \
      "data.tokenized.overwrite=${OVERWRITE_TOKENIZED}" \
      "data.tokenized.append_eos=true"
else
  echo "Unknown TOKENIZE_MODE=${TOKENIZE_MODE}" >&2
  exit 2
fi

echo "[$(date -Is)] Finished mirror-probe render/tokenize"
