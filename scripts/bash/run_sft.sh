#!/usr/bin/env bash
set -euo pipefail

# QA-only SFT launcher. Edit variables below, then run:
#   bash scripts/bash/run_sft.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/sft_qa_smollm2_135m.yaml"

EXPERIMENT_NAME="smollm2-135m-sft-reverse-qa-v2-50epoch"
VISIBLE_DEVICES="0,1,2,3"
MIXED_PRECISION="auto"

BASE_CHECKPOINT_STEP="/data3/yizhou/projects/safe-pretrain/outputs/smollm2-135m-scratch-0p3b-1epoch-bs512-synthetic_world_4096effects_8192causes_0.5restricted_2arity_strict_wo_overlap/pretrain/0.25reverse_0.99train_composition_v1/checkpoints/step-0001179"
BASE_CHECKPOINT="${BASE_CHECKPOINT_STEP}/hf_model"
SFT_DATASET_ROOT="/data3/yizhou/projects/safe-pretrain/data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_2arity_strict_wo_overlap/sft/qa_1ex_0.8train_0.1val_composition_v1_0.5restrict-train"
TRAIN_FILE="${SFT_DATASET_ROOT}/sft_train.jsonl"
VALIDATION_FILE="${SFT_DATASET_ROOT}/sft_validation.jsonl"
CHAT_TEMPLATE_PATH="${SFT_DATASET_ROOT}/chat_template.jinja"

MAX_LENGTH=256
PACKING=false

NUM_TRAIN_EPOCHS=50
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE="2.0e-5"
LOGGING_STEPS=1
EVAL_STEPS=100
SAVE_STEPS=200
ACCURACY_EVAL_ENABLED=true
ACCURACY_TRAIN_EXAMPLES=512
ACCURACY_VAL_EXAMPLES=2048
ACCURACY_BATCH_SIZE=64
ACCURACY_MAX_NEW_TOKENS=32
WANDB_ENABLED=true

export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

if [[ ! -f "${BASE_CHECKPOINT}/config.json" || ! -f "${BASE_CHECKPOINT}/tokenizer.json" ]]; then
  echo "Expected HuggingFace model/tokenizer files under: ${BASE_CHECKPOINT}" >&2
  echo "Set BASE_CHECKPOINT_STEP to a checkpoint step directory containing hf_model/." >&2
  exit 1
fi

python scripts/python/launch_sft.py \
  --config "${CONFIG}" \
  project.experiment_name="${EXPERIMENT_NAME}" \
  runtime.visible_devices="${VISIBLE_DEVICES}" \
  runtime.mixed_precision="${MIXED_PRECISION}" \
  model.base_checkpoint="${BASE_CHECKPOINT}" \
  data.dataset_root="${SFT_DATASET_ROOT}" \
  data.train_file="${TRAIN_FILE}" \
  data.validation_file="${VALIDATION_FILE}" \
  data.chat_template_path="${CHAT_TEMPLATE_PATH}" \
  data.max_length="${MAX_LENGTH}" \
  data.packing="${PACKING}" \
  train.num_train_epochs="${NUM_TRAIN_EPOCHS}" \
  train.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  train.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  train.learning_rate="${LEARNING_RATE}" \
  train.logging_steps="${LOGGING_STEPS}" \
  train.eval_steps="${EVAL_STEPS}" \
  train.save_steps="${SAVE_STEPS}" \
  accuracy_eval.enabled="${ACCURACY_EVAL_ENABLED}" \
  accuracy_eval.train_examples="${ACCURACY_TRAIN_EXAMPLES}" \
  accuracy_eval.val_examples="${ACCURACY_VAL_EXAMPLES}" \
  accuracy_eval.batch_size="${ACCURACY_BATCH_SIZE}" \
  accuracy_eval.max_new_tokens="${ACCURACY_MAX_NEW_TOKENS}" \
  wandb.enabled="${WANDB_ENABLED}"
