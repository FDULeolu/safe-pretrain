#!/usr/bin/env bash
set -euo pipefail

# One-entry experiment launcher for the current synthetic pipeline.
#
# Typical usage:
#   FAMILY=ocr CHAT_TEMPLATE=plain SFT_REPEAT_FORWARD_IDENTITY=6 bash scripts/bash/run_experiment_pipeline.sh
#
# Stages can be disabled independently:
#   RUN_DATA=true RUN_PRETRAIN=false RUN_SFT=false bash scripts/bash/run_experiment_pipeline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET_CONFIG="${DATASET_CONFIG:-configs/synthetic_dataset.yaml}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/pretrain_a6000_smollm2_135m.yaml}"
SFT_CONFIG="${SFT_CONFIG:-configs/sft_qa_smollm2_135m.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

FAMILY="${FAMILY:-ocr}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-plain}"
WORLD_NAME="${WORLD_NAME:-w1024-c2048-a1-r10-dic-strict}"
NUM_EFFECTS="${NUM_EFFECTS:-1024}"
NUM_CAUSES="${NUM_CAUSES:-2048}"
ARITY="${ARITY:-1}"
RESTRICTED_FRACTION="${RESTRICTED_FRACTION:-0.1}"
CAUSE_VOCAB_SOURCE="${CAUSE_VOCAB_SOURCE:-tokenizer_english_words}"
EFFECT_VOCAB_SOURCE="${EFFECT_VOCAB_SOURCE:-tokenizer_english_words}"
USE_FAMILIES="${USE_FAMILIES:-false}"
UNIQUE_RECIPE_PER_EFFECT="${UNIQUE_RECIPE_PER_EFFECT:-true}"
UNIQUE_CAUSE_TUPLE="${UNIQUE_CAUSE_TUPLE:-true}"
ALLOW_DUPLICATE_CAUSE_IN_RECIPE="${ALLOW_DUPLICATE_CAUSE_IN_RECIPE:-false}"
CAUSE_FREQUENCY_BALANCE="${CAUSE_FREQUENCY_BALANCE:-true}"

SFT_REPEAT_FORWARD="${SFT_REPEAT_FORWARD:-1}"
SFT_REPEAT_REVERSE="${SFT_REPEAT_REVERSE:-1}"
SFT_REPEAT_IDENTITY="${SFT_REPEAT_IDENTITY:-1}"
SFT_REPEAT_FORWARD_IDENTITY="${SFT_REPEAT_FORWARD_IDENTITY:-6}"
SFT_REPEAT_FORWARD_REVERSE="${SFT_REPEAT_FORWARD_REVERSE:-1}"
SFT_REPEAT_PREVENTION="${SFT_REPEAT_PREVENTION:-1}"

DEFAULT_EXPERIMENT_NAME="${FAMILY}_${WORLD_NAME}_${CHAT_TEMPLATE}"
if [[ "${FAMILY}" == "ocr" || "${FAMILY}" == "ocr_linear" ]]; then
  DEFAULT_EXPERIMENT_NAME="${DEFAULT_EXPERIMENT_NAME}_fi${SFT_REPEAT_FORWARD_IDENTITY}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-data/experiments/${EXPERIMENT_NAME}}"
OVERWRITE_DATA="${OVERWRITE_DATA:-false}"

TARGET_TOKENS="${TARGET_TOKENS:-300000000}"
TARGET_RECORDS="${TARGET_RECORDS:-null}"
TOKEN_ESTIMATE_RECORDS="${TOKEN_ESTIMATE_RECORDS:-4096}"
TOKEN_ESTIMATE_BATCH_SIZE="${TOKEN_ESTIMATE_BATCH_SIZE:-1024}"
ESTIMATED_TOKENS_PER_RECORD="${ESTIMATED_TOKENS_PER_RECORD:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-16384}"
GENERATE_PRETRAIN="${GENERATE_PRETRAIN:-true}"
GENERATE_TOKENIZED="${GENERATE_TOKENIZED:-true}"
GENERATE_SFT="${GENERATE_SFT:-true}"
PRETRAIN_EVAL_MEMORY_RECORDS="${PRETRAIN_EVAL_MEMORY_RECORDS:-4096}"
PRETRAIN_EVAL_TEMPLATE_EXAMPLES_PER_PATTERN="${PRETRAIN_EVAL_TEMPLATE_EXAMPLES_PER_PATTERN:-1}"

SFT_INCLUDE_VALIDATION="${SFT_INCLUDE_VALIDATION:-false}"
SFT_VALIDATION_FRACTION="${SFT_VALIDATION_FRACTION:-0.0}"
SFT_TEST_RELATION_FRACTION="${SFT_TEST_RELATION_FRACTION:-0.1}"
SFT_EVAL_INCLUDE_MEMORY="${SFT_EVAL_INCLUDE_MEMORY:-false}"
SFT_EVAL_TEMPLATE_EXAMPLES_PER_PATTERN="${SFT_EVAL_TEMPLATE_EXAMPLES_PER_PATTERN:-1}"
SFT_EVAL_ATTACK_EXAMPLES_PER_PATTERN="${SFT_EVAL_ATTACK_EXAMPLES_PER_PATTERN:-1}"

RUN_DATA="${RUN_DATA:-true}"
RUN_PRETRAIN="${RUN_PRETRAIN:-false}"
RUN_SFT="${RUN_SFT:-false}"
RUN_EVAL_PRETRAIN="${RUN_EVAL_PRETRAIN:-${RUN_PRETRAIN}}"
RUN_EVAL_SFT="${RUN_EVAL_SFT:-${RUN_SFT}}"

VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-null}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"
PRETRAIN_PER_DEVICE_BATCH_SIZE="${PRETRAIN_PER_DEVICE_BATCH_SIZE:-64}"
PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-1}"
PRETRAIN_LR="${PRETRAIN_LR:-3.0e-4}"
PRETRAIN_WD="${PRETRAIN_WD:-1.0}"
PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-250}"
PRETRAIN_EVAL_STEPS="${PRETRAIN_EVAL_STEPS:-100}"
PRETRAIN_KEEP_LAST="${PRETRAIN_KEEP_LAST:-1}"
PRETRAIN_AUTO_RESUME="${PRETRAIN_AUTO_RESUME:-true}"
PRETRAIN_SKIP_IF_COMPLETE="${PRETRAIN_SKIP_IF_COMPLETE:-true}"

SFT_BASE_CHECKPOINT="${SFT_BASE_CHECKPOINT:-}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-5000}"
SFT_PER_DEVICE_BATCH_SIZE="${SFT_PER_DEVICE_BATCH_SIZE:-16}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-1}"
SFT_LR="${SFT_LR:-1.0e-4}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-256}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-500}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-null}"
SFT_ASSISTANT_ONLY_LOSS="${SFT_ASSISTANT_ONLY_LOSS:-true}"
SFT_COMPLETION_ONLY_LOSS="${SFT_COMPLETION_ONLY_LOSS:-false}"
SFT_ACCURACY_EVAL="${SFT_ACCURACY_EVAL:-false}"
SFT_AUTO_RESUME="${SFT_AUTO_RESUME:-true}"
SFT_SKIP_IF_COMPLETE="${SFT_SKIP_IF_COMPLETE:-true}"

EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_DTYPE="${EVAL_DTYPE:-auto}"
PRETRAIN_EVAL_BATCH_SIZE="${PRETRAIN_EVAL_BATCH_SIZE:-64}"
PRETRAIN_EVAL_MAX_NEW_TOKENS="${PRETRAIN_EVAL_MAX_NEW_TOKENS:-32}"
PRETRAIN_EVAL_MAX_EXAMPLES="${PRETRAIN_EVAL_MAX_EXAMPLES:-null}"
SFT_EVAL_BATCH_SIZE="${SFT_EVAL_BATCH_SIZE:-64}"
SFT_EVAL_MAX_NEW_TOKENS="${SFT_EVAL_MAX_NEW_TOKENS:-32}"
SFT_EVAL_MAX_EXAMPLES="${SFT_EVAL_MAX_EXAMPLES:-null}"
SFT_EVAL_CHECKPOINTS="${SFT_EVAL_CHECKPOINTS:-all}"

export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

WORLD_DIR="${WORLD_DIR:-data/worlds/${WORLD_NAME}}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${EXPERIMENT_ROOT}/pretrain}"
TOKENIZED_DIR="${TOKENIZED_DIR:-${PRETRAIN_DIR}/tokenized/bs${BLOCK_SIZE}}"
SFT_DIR="${SFT_DIR:-${EXPERIMENT_ROOT}/sft_${CHAT_TEMPLATE}}"
PRETRAIN_EXPERIMENT="${PRETRAIN_EXPERIMENT:-pt-${EXPERIMENT_NAME}-lr${PRETRAIN_LR}-wd${PRETRAIN_WD}}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-outputs/${PRETRAIN_EXPERIMENT}}"
PRETRAIN_EVAL_OUTPUT_DIR="${PRETRAIN_EVAL_OUTPUT_DIR:-${PRETRAIN_OUTPUT_DIR}/eval/pretrain_completion}"
SFT_EXPERIMENT="${SFT_EXPERIMENT:-sft-${EXPERIMENT_NAME}-lr${SFT_LR}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-outputs/${SFT_EXPERIMENT}}"
SFT_EVAL_OUTPUT_ROOT="${SFT_EVAL_OUTPUT_ROOT:-${SFT_OUTPUT_DIR}/eval/sft_qa}"

SFT_REPEAT_OVERRIDES=()
case "${FAMILY}" in
  vanilla | mirror)
    SFT_REPEAT_OVERRIDES=(
      "sft.pattern_repeats.forward=${SFT_REPEAT_FORWARD}"
      "sft.pattern_repeats.reverse=${SFT_REPEAT_REVERSE}"
    )
    ;;
  ocr | ocr_linear)
    SFT_REPEAT_OVERRIDES=(
      "sft.pattern_repeats.reverse=${SFT_REPEAT_REVERSE}"
      "sft.pattern_repeats.identity=${SFT_REPEAT_IDENTITY}"
      "sft.pattern_repeats.forward_identity=${SFT_REPEAT_FORWARD_IDENTITY}"
      "sft.pattern_repeats.forward_reverse=${SFT_REPEAT_FORWARD_REVERSE}"
    )
    ;;
  prevention)
    SFT_REPEAT_OVERRIDES=(
      "sft.pattern_repeats.forward=${SFT_REPEAT_FORWARD}"
      "sft.pattern_repeats.reverse=${SFT_REPEAT_REVERSE}"
      "sft.pattern_repeats.prevention=${SFT_REPEAT_PREVENTION}"
    )
    ;;
  *)
    echo "Unsupported FAMILY=${FAMILY}" >&2
    exit 2
    ;;
esac

DATA_OVERRIDES=(
  "experiment.name=${EXPERIMENT_NAME}"
  "experiment.root=${EXPERIMENT_ROOT}"
  "experiment.overwrite=${OVERWRITE_DATA}"
  "dataset.family=${FAMILY}"
  "world.name=${WORLD_NAME}"
  "world.path=${WORLD_DIR}"
  "world.overwrite=${OVERWRITE_DATA}"
  "world.num_effects=${NUM_EFFECTS}"
  "world.num_causes=${NUM_CAUSES}"
  "world.recipe_arity=${ARITY}"
  "world.partition.restricted_fraction=${RESTRICTED_FRACTION}"
  "world.surface.cause_vocab_source=${CAUSE_VOCAB_SOURCE}"
  "world.surface.effect_vocab_source=${EFFECT_VOCAB_SOURCE}"
  "world.surface.use_families=${USE_FAMILIES}"
  "world.relations.unique_recipe_per_effect=${UNIQUE_RECIPE_PER_EFFECT}"
  "world.relations.unique_cause_tuple=${UNIQUE_CAUSE_TUPLE}"
  "world.relations.allow_duplicate_cause_in_recipe=${ALLOW_DUPLICATE_CAUSE_IN_RECIPE}"
  "world.relations.cause_frequency_balance=${CAUSE_FREQUENCY_BALANCE}"
  "pretrain.enabled=${GENERATE_PRETRAIN}"
  "pretrain.output_dir=${PRETRAIN_DIR}"
  "pretrain.overwrite=${OVERWRITE_DATA}"
  "pretrain.target_tokens=${TARGET_TOKENS}"
  "pretrain.target_records=${TARGET_RECORDS}"
  "pretrain.token_estimate_records=${TOKEN_ESTIMATE_RECORDS}"
  "pretrain.token_estimate_batch_size=${TOKEN_ESTIMATE_BATCH_SIZE}"
  "pretrain.estimated_tokens_per_record=${ESTIMATED_TOKENS_PER_RECORD}"
  "pretrain.eval.memory_records=${PRETRAIN_EVAL_MEMORY_RECORDS}"
  "pretrain.eval.template_examples_per_pattern=${PRETRAIN_EVAL_TEMPLATE_EXAMPLES_PER_PATTERN}"
  "tokenize.enabled=${GENERATE_TOKENIZED}"
  "tokenize.output_dir=${TOKENIZED_DIR}"
  "tokenize.overwrite=${OVERWRITE_DATA}"
  "tokenize.block_size=${BLOCK_SIZE}"
  "tokenize.batch_size=${TOKENIZE_BATCH_SIZE}"
  "sft.enabled=${GENERATE_SFT}"
  "sft.output_dir=${SFT_DIR}"
  "sft.overwrite=${OVERWRITE_DATA}"
  "sft.chat_template=${CHAT_TEMPLATE}"
  "sft.include_validation=${SFT_INCLUDE_VALIDATION}"
  "sft.validation_fraction=${SFT_VALIDATION_FRACTION}"
  "sft.test_relation_fraction=${SFT_TEST_RELATION_FRACTION}"
  "sft.eval.include_memory=${SFT_EVAL_INCLUDE_MEMORY}"
  "sft.eval.template_examples_per_pattern=${SFT_EVAL_TEMPLATE_EXAMPLES_PER_PATTERN}"
  "sft.eval.attack_examples_per_pattern=${SFT_EVAL_ATTACK_EXAMPLES_PER_PATTERN}"
  "${SFT_REPEAT_OVERRIDES[@]}"
)

PRETRAIN_OVERRIDES=(
  "project.experiment_name=${PRETRAIN_EXPERIMENT}"
  "project.output_dir=${PRETRAIN_OUTPUT_DIR}"
  "runtime.visible_devices=${VISIBLE_DEVICES}"
  "runtime.main_process_port=${MAIN_PROCESS_PORT}"
  "runtime.mixed_precision=${MIXED_PRECISION}"
  "data.tokenized.path=${TOKENIZED_DIR}"
  "dataloader.per_device_batch_size=${PRETRAIN_PER_DEVICE_BATCH_SIZE}"
  "train.gradient_accumulation_steps=${PRETRAIN_GRAD_ACCUM}"
  "train.learning_rate=${PRETRAIN_LR}"
  "train.weight_decay=${PRETRAIN_WD}"
  "train.eval_every_steps=${PRETRAIN_EVAL_STEPS}"
  "checkpoint.save_every_steps=${PRETRAIN_SAVE_STEPS}"
  "checkpoint.keep_last=${PRETRAIN_KEEP_LAST}"
)

run_dataset_stage() {
  local stage="$1"
  shift || true
  "${PYTHON_BIN}" scripts/python/build_synthetic_dataset.py \
    --config "${DATASET_CONFIG}" \
    --stage "${stage}" \
    "${DATA_OVERRIDES[@]}" \
    "$@"
}

PRETRAIN_CHECKPOINT_STATUS_OUTPUT=""

pretrain_checkpoint_status() {
  local output=""
  local status=0
  output=$("${PYTHON_BIN}" scripts/python/check_pretrain_checkpoint.py \
    --config "${PRETRAIN_CONFIG}" \
    "${PRETRAIN_OVERRIDES[@]}" 2>&1) || status=$?
  PRETRAIN_CHECKPOINT_STATUS_OUTPUT="${output}"
  return "${status}"
}

pretrain_checkpoint_field() {
  local field="$1"
  "${PYTHON_BIN}" scripts/python/check_pretrain_checkpoint.py \
    --config "${PRETRAIN_CONFIG}" \
    --field "${field}" \
    "${PRETRAIN_OVERRIDES[@]}"
}

resolve_completed_pretrain_hf_model() {
  local status=0
  set +e
  pretrain_checkpoint_status
  status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    pretrain_checkpoint_field hf_model
    return 0
  fi
  echo "${PRETRAIN_CHECKPOINT_STATUS_OUTPUT}" >&2
  if [[ "${status}" -eq 3 ]]; then
    echo "Pretrain checkpoint exists but is not complete; finish pretraining or set SFT_BASE_CHECKPOINT explicitly." >&2
  elif [[ "${status}" -eq 1 ]]; then
    echo "No matching completed pretrain checkpoint found for ${PRETRAIN_EXPERIMENT}." >&2
  elif [[ "${status}" -eq 2 ]]; then
    echo "Existing pretrain checkpoint config does not match the current pretrain config." >&2
  fi
  return "${status}"
}

sft_final_model_complete() {
  local model_dir="${SFT_OUTPUT_DIR}/final_model"
  [[ -f "${SFT_OUTPUT_DIR}/sft_config.yaml" ]] || return 1
  [[ -d "${model_dir}" && -f "${model_dir}/config.json" ]] || return 1
  [[ -n "$(find "${model_dir}" -maxdepth 1 -type f \( \
    -name 'model.safetensors' -o \
    -name 'model.safetensors.index.json' -o \
    -name 'pytorch_model.bin' -o \
    -name 'pytorch_model.bin.index.json' \
  \) -print -quit)" ]]
}

sft_latest_checkpoint() {
  local latest=""
  if [[ ! -d "${SFT_OUTPUT_DIR}" ]]; then
    return 1
  fi
  latest="$(find "${SFT_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  echo "${latest}"
}

world_ready() {
  artifact_ready world
}

pretrain_ready() {
  artifact_ready pretrain
}

tokenized_ready() {
  artifact_ready tokenize
}

sft_ready() {
  artifact_ready sft
}

artifact_ready() {
  local stage="$1"
  local output=""
  local status=0
  output=$("${PYTHON_BIN}" scripts/python/check_synthetic_artifact.py \
    --config "${DATASET_CONFIG}" \
    --stage "${stage}" \
    "${DATA_OVERRIDES[@]}" 2>&1) || status=$?
  if [[ "${status}" -eq 0 ]]; then
    return 0
  fi
  if [[ "${status}" -eq 1 ]]; then
    return 1
  fi
  echo "${output}" >&2
  if [[ "${status}" -eq 2 ]]; then
    echo "Configured ${stage} artifact does not match the current config." >&2
    echo "Use a new EXPERIMENT_NAME/WORLD_NAME, set OVERWRITE_DATA=true for an individual experiment, or inspect before overwriting." >&2
  fi
  exit "${status}"
}

ensure_dataset_artifacts() {
  local world_built="false"
  local pretrain_built="false"

  if [[ "${OVERWRITE_DATA}" == "true" ]]; then
    echo "overwrite_data=true; rebuilding configured dataset artifacts"
  fi

  if world_ready && [[ "${OVERWRITE_DATA}" != "true" ]]; then
    echo "world ready: ${WORLD_DIR}"
  else
    echo "building world: ${WORLD_DIR}"
    run_dataset_stage world "world.overwrite=true"
    world_built="true"
  fi

  if [[ "${GENERATE_PRETRAIN}" == "true" ]]; then
    if pretrain_ready && [[ "${OVERWRITE_DATA}" != "true" ]] && [[ "${world_built}" != "true" ]]; then
      echo "pretrain ready: ${PRETRAIN_DIR}"
    else
      echo "building pretrain: ${PRETRAIN_DIR}"
      run_dataset_stage pretrain "pretrain.overwrite=true"
      pretrain_built="true"
    fi
  fi

  if [[ "${GENERATE_TOKENIZED}" == "true" ]]; then
    if ! pretrain_ready; then
      echo "Tokenization requires pretrain JSONL, but it is missing: ${PRETRAIN_DIR}" >&2
      exit 2
    fi
    if tokenized_ready && [[ "${OVERWRITE_DATA}" != "true" ]] && [[ "${pretrain_built}" != "true" ]]; then
      echo "tokenized pretrain ready: ${TOKENIZED_DIR}"
    else
      echo "building tokenized pretrain: ${TOKENIZED_DIR}"
      run_dataset_stage tokenize "tokenize.overwrite=true"
    fi
  fi

  if [[ "${GENERATE_SFT}" == "true" ]]; then
    if sft_ready && [[ "${OVERWRITE_DATA}" != "true" ]] && [[ "${world_built}" != "true" ]]; then
      echo "sft ready: ${SFT_DIR}"
    else
      echo "building sft: ${SFT_DIR}"
      run_dataset_stage sft "sft.overwrite=true"
    fi
  fi
}

echo "experiment: ${EXPERIMENT_NAME}"
echo "family: ${FAMILY}"
echo "chat_template: ${CHAT_TEMPLATE}"
echo "experiment_root: ${EXPERIMENT_ROOT}"
echo "pretrain_experiment: ${PRETRAIN_EXPERIMENT}"
echo "pretrain_output_dir: ${PRETRAIN_OUTPUT_DIR}"
echo "sft_experiment: ${SFT_EXPERIMENT}"
echo "sft_output_dir: ${SFT_OUTPUT_DIR}"
echo "run_data=${RUN_DATA} run_pretrain=${RUN_PRETRAIN} run_sft=${RUN_SFT} run_eval_pretrain=${RUN_EVAL_PRETRAIN} run_eval_sft=${RUN_EVAL_SFT}"

if [[ "${RUN_DATA}" == "true" ]]; then
  ensure_dataset_artifacts
fi

if [[ "${RUN_PRETRAIN}" == "true" ]]; then
  if ! tokenized_ready; then
    echo "Tokenized pretrain data is missing: ${TOKENIZED_DIR}" >&2
    echo "Set RUN_DATA=true and GENERATE_TOKENIZED=true, or point TOKENIZED_DIR to an existing dataset." >&2
    exit 2
  fi
  SHOULD_RUN_PRETRAIN="true"
  PRETRAIN_RESUME_OVERRIDE=()
  if [[ "${PRETRAIN_AUTO_RESUME}" == "true" ]]; then
    set +e
    pretrain_checkpoint_status
    CHECKPOINT_STATUS=$?
    set -e
    if [[ "${CHECKPOINT_STATUS}" -eq 0 ]]; then
      PRETRAIN_LATEST_HF_MODEL="$(pretrain_checkpoint_field hf_model)"
      if [[ -z "${SFT_BASE_CHECKPOINT}" ]]; then
        SFT_BASE_CHECKPOINT="${PRETRAIN_LATEST_HF_MODEL}"
      fi
      if [[ "${PRETRAIN_SKIP_IF_COMPLETE}" == "true" ]]; then
        echo "pretrain checkpoint complete; skipping pretrain: ${PRETRAIN_LATEST_HF_MODEL}"
        SHOULD_RUN_PRETRAIN="false"
      else
        PRETRAIN_RESUME_DIR="$(pretrain_checkpoint_field resume_dir)"
        PRETRAIN_RESUME_OVERRIDE=("checkpoint.resume_from=${PRETRAIN_RESUME_DIR}")
        echo "pretrain checkpoint complete; launching with resume_from=${PRETRAIN_RESUME_DIR}"
      fi
    elif [[ "${CHECKPOINT_STATUS}" -eq 3 ]]; then
      PRETRAIN_RESUME_DIR="$(pretrain_checkpoint_field resume_dir)"
      PRETRAIN_RESUME_OVERRIDE=("checkpoint.resume_from=${PRETRAIN_RESUME_DIR}")
      echo "resuming incomplete pretrain checkpoint: ${PRETRAIN_RESUME_DIR}"
    elif [[ "${CHECKPOINT_STATUS}" -eq 1 ]]; then
      echo "no pretrain checkpoint found; starting from scratch: ${PRETRAIN_OUTPUT_DIR}"
    else
      echo "${PRETRAIN_CHECKPOINT_STATUS_OUTPUT}" >&2
      if [[ "${CHECKPOINT_STATUS}" -eq 2 ]]; then
        echo "Existing pretrain checkpoint config does not match the current pretrain config." >&2
        echo "Use a new PRETRAIN_EXPERIMENT/PRETRAIN_OUTPUT_DIR, or inspect the existing checkpoint before overwriting." >&2
      fi
      exit "${CHECKPOINT_STATUS}"
    fi
  fi
  if [[ "${SHOULD_RUN_PRETRAIN}" == "true" ]]; then
    "${PYTHON_BIN}" scripts/python/launch_pretrain.py \
      --config "${PRETRAIN_CONFIG}" \
      "${PRETRAIN_OVERRIDES[@]}" \
      "${PRETRAIN_RESUME_OVERRIDE[@]}"
  fi
fi

if [[ "${RUN_EVAL_PRETRAIN}" == "true" ]]; then
  if ! pretrain_ready; then
    echo "Pretrain eval data is missing: ${PRETRAIN_DIR}" >&2
    echo "Set RUN_DATA=true and GENERATE_PRETRAIN=true, or point PRETRAIN_DIR to an existing dataset." >&2
    exit 2
  fi
  PRETRAIN_EVAL_MODEL="${PRETRAIN_EVAL_MODEL:-}"
  if [[ -z "${PRETRAIN_EVAL_MODEL}" ]]; then
    PRETRAIN_EVAL_MODEL="$(resolve_completed_pretrain_hf_model)"
  fi
  "${PYTHON_BIN}" scripts/python/eval_pretrain_completion.py \
    --model "${PRETRAIN_EVAL_MODEL}" \
    --pretrain-dir "${PRETRAIN_DIR}" \
    --output-dir "${PRETRAIN_EVAL_OUTPUT_DIR}" \
    --max-examples "${PRETRAIN_EVAL_MAX_EXAMPLES}" \
    --device "${EVAL_DEVICE}" \
    --dtype "${EVAL_DTYPE}" \
    --batch-size "${PRETRAIN_EVAL_BATCH_SIZE}" \
    --max-new-tokens "${PRETRAIN_EVAL_MAX_NEW_TOKENS}"
fi

if [[ "${RUN_SFT}" == "true" ]]; then
  if ! sft_ready; then
    echo "SFT data is missing: ${SFT_DIR}" >&2
    echo "Set RUN_DATA=true and GENERATE_SFT=true, or point SFT_DIR to an existing dataset." >&2
    exit 2
  fi
  if [[ -z "${SFT_BASE_CHECKPOINT}" ]]; then
    if [[ "${PRETRAIN_AUTO_RESUME}" == "true" ]]; then
      SFT_BASE_CHECKPOINT="$(resolve_completed_pretrain_hf_model)"
    else
      echo "SFT_BASE_CHECKPOINT must point to a pretrain hf_model directory when RUN_SFT=true" >&2
      exit 2
    fi
  fi
  SHOULD_RUN_SFT="true"
  SFT_RESUME_OVERRIDE=()
  if [[ "${SFT_SKIP_IF_COMPLETE}" == "true" ]] && sft_final_model_complete; then
    echo "sft final_model complete; skipping SFT: ${SFT_OUTPUT_DIR}/final_model"
    SHOULD_RUN_SFT="false"
  elif [[ "${SFT_AUTO_RESUME}" == "true" ]]; then
    SFT_RESUME_DIR="$(sft_latest_checkpoint || true)"
    if [[ -n "${SFT_RESUME_DIR}" ]]; then
      SFT_RESUME_OVERRIDE=("checkpoint.resume_from=${SFT_RESUME_DIR}")
      echo "resuming incomplete SFT checkpoint: ${SFT_RESUME_DIR}"
    fi
  fi
  if [[ "${SHOULD_RUN_SFT}" == "true" ]]; then
    "${PYTHON_BIN}" scripts/python/launch_sft.py \
      --config "${SFT_CONFIG}" \
      "project.experiment_name=${SFT_EXPERIMENT}" \
      "project.output_dir=${SFT_OUTPUT_DIR}" \
      "runtime.visible_devices=${VISIBLE_DEVICES}" \
      "runtime.main_process_port=${MAIN_PROCESS_PORT}" \
      "runtime.mixed_precision=${MIXED_PRECISION}" \
      "model.base_checkpoint=${SFT_BASE_CHECKPOINT}" \
      "data.dataset_root=${SFT_DIR}" \
      "data.train_file=${SFT_DIR}/sft_train.jsonl" \
      "data.validation_file=${SFT_DIR}/sft_validation.jsonl" \
      "data.attack_file=${SFT_DIR}/eval_attack.jsonl" \
      "data.chat_template_path=${SFT_DIR}/chat_template.jinja" \
      "data.max_length=${SFT_MAX_LENGTH}" \
      "data.packing=false" \
      "data.completion_only_loss=${SFT_COMPLETION_ONLY_LOSS}" \
      "data.assistant_only_loss=${SFT_ASSISTANT_ONLY_LOSS}" \
      "train.max_steps=${SFT_MAX_STEPS}" \
      "train.num_train_epochs=9999" \
      "train.per_device_train_batch_size=${SFT_PER_DEVICE_BATCH_SIZE}" \
      "train.gradient_accumulation_steps=${SFT_GRAD_ACCUM}" \
      "train.learning_rate=${SFT_LR}" \
      "train.save_steps=${SFT_SAVE_STEPS}" \
      "train.save_total_limit=${SFT_SAVE_TOTAL_LIMIT}" \
      "accuracy_eval.enabled=${SFT_ACCURACY_EVAL}" \
      "${SFT_RESUME_OVERRIDE[@]}"
  fi
fi

if [[ "${RUN_EVAL_SFT}" == "true" ]]; then
  if ! sft_ready; then
    echo "SFT eval data is missing: ${SFT_DIR}" >&2
    echo "Set RUN_DATA=true and GENERATE_SFT=true, or point SFT_DIR to an existing dataset." >&2
    exit 2
  fi
  if [[ ! -d "${SFT_OUTPUT_DIR}" ]]; then
    echo "SFT output directory is missing: ${SFT_OUTPUT_DIR}" >&2
    exit 2
  fi
  SFT_EVAL_MODELS=()
  if [[ "${SFT_EVAL_CHECKPOINTS}" == "all" ]]; then
    for checkpoint_dir in "${SFT_OUTPUT_DIR}"/checkpoint-*; do
      if [[ -d "${checkpoint_dir}" ]]; then
        SFT_EVAL_MODELS+=("${checkpoint_dir}")
      fi
    done
    if [[ -d "${SFT_OUTPUT_DIR}/final_model" ]]; then
      SFT_EVAL_MODELS+=("${SFT_OUTPUT_DIR}/final_model")
    fi
  elif [[ "${SFT_EVAL_CHECKPOINTS}" == "final" ]]; then
    SFT_EVAL_MODELS+=("${SFT_OUTPUT_DIR}/final_model")
  else
    IFS=',' read -r -a SFT_EVAL_MODELS <<< "${SFT_EVAL_CHECKPOINTS}"
  fi
  if [[ "${#SFT_EVAL_MODELS[@]}" -eq 0 ]]; then
    echo "No SFT checkpoints found for eval under ${SFT_OUTPUT_DIR}" >&2
    exit 2
  fi
  for model_dir in "${SFT_EVAL_MODELS[@]}"; do
    if [[ ! -d "${model_dir}" ]]; then
      echo "SFT eval model directory is missing: ${model_dir}" >&2
      exit 2
    fi
    model_name="$(basename "${model_dir}")"
    "${PYTHON_BIN}" scripts/python/eval_sft_qa.py \
      --model "${model_dir}" \
      --sft-dir "${SFT_DIR}" \
      --output-dir "${SFT_EVAL_OUTPUT_ROOT}/${model_name}" \
      --max-examples "${SFT_EVAL_MAX_EXAMPLES}" \
      --device "${EVAL_DEVICE}" \
      --dtype "${EVAL_DTYPE}" \
      --batch-size "${SFT_EVAL_BATCH_SIZE}" \
      --max-new-tokens "${SFT_EVAL_MAX_NEW_TOKENS}"
  done
fi
