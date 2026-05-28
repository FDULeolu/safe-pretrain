#!/usr/bin/env bash
set -euo pipefail

# QA-only SFT launcher. Edit variables below, then run:
#   bash scripts/bash/run_sft.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/sft_qa_smollm2_135m.yaml"

EXPERIMENT_NAME="smollm2-135m-sft-reverse-qa-v1"
VISIBLE_DEVICES="0,1,2,3"
MIXED_PRECISION="auto"

BASE_CHECKPOINT="outputs/smollm2-135m-scratch-0p3b-1epoch-bs512/checkpoints/step-0001209/hf_model"
TRAIN_FILE="data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap/sft/reverse_qa/sft_train.jsonl"
VALIDATION_FILE="data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap/sft/reverse_qa/sft_validation.jsonl"

MAX_LENGTH=256
PACKING=false

NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE="2.0e-5"
LOGGING_STEPS=10
EVAL_STEPS=100
SAVE_STEPS=500
ACCURACY_EVAL_ENABLED=true
ACCURACY_TRAIN_EXAMPLES=512
ACCURACY_VAL_EXAMPLES=2048
ACCURACY_BATCH_SIZE=64
ACCURACY_MAX_NEW_TOKENS=32
WANDB_ENABLED=true

export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

python scripts/python/launch_sft.py \
  --config "${CONFIG}" \
  project.experiment_name="${EXPERIMENT_NAME}" \
  runtime.visible_devices="${VISIBLE_DEVICES}" \
  runtime.mixed_precision="${MIXED_PRECISION}" \
  model.base_checkpoint="${BASE_CHECKPOINT}" \
  data.train_file="${TRAIN_FILE}" \
  data.validation_file="${VALIDATION_FILE}" \
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
