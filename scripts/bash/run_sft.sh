#!/usr/bin/env bash
set -euo pipefail

# QA-only SFT launcher. Edit variables below, then run:
#   bash scripts/bash/run_sft.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/sft_qa_smollm2_135m.yaml"

VISIBLE_DEVICES="0,1,2,3"
MIXED_PRECISION="auto"

BASE_CHECKPOINT_STEP="/data3/yizhou/projects/safe-pretrain/outputs/smollm2-135m-scratch-0p3b-1epoch-bs512-synthetic_world_1024effects_2048causes_0.1restricted_2arity_strict_wo_overlap_dic-words/pretrain/0.45reverse_0.99train_composition_v1_pretrain_descriptive_v2_small-batch/checkpoints/step-0002299"
BASE_CHECKPOINT="${BASE_CHECKPOINT_STEP}/hf_model"
SFT_DATASET_ROOT="/data3/yizhou/projects/safe-pretrain/data/worlds/synthetic_world_1024effects_2048causes_0.1restricted_2arity_strict_wo_overlap_dic-words/sft/qa_32ex_0.8train_0.1val_composition_v1_0.5restrict-train"
TRAIN_FILE="${SFT_DATASET_ROOT}/sft_train.jsonl"
VALIDATION_FILE="${SFT_DATASET_ROOT}/sft_validation.jsonl"
CHAT_TEMPLATE_PATH="${SFT_DATASET_ROOT}/chat_template.jinja"

MAX_LENGTH=256
PACKING=false
COMPLETION_ONLY_LOSS=false
ASSISTANT_ONLY_LOSS=false

NUM_TRAIN_EPOCHS=9999
MAX_STEPS=5000
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE="1.0e-4"
LOGGING_STEPS=1
EVAL_STEPS=100
SAVE_STEPS=200
ACCURACY_EVAL_ENABLED=true
ACCURACY_TRAIN_EXAMPLES=512
ACCURACY_VAL_EXAMPLES=410
ACCURACY_BATCH_SIZE=64
ACCURACY_MAX_NEW_TOKENS=32
WANDB_ENABLED=true

EXPERIMENT_NAME="smollm2-135m-sft-reverse-qa-32ex-fullseq-${MAX_STEPS}steps-${LEARNING_RATE}lr-${PER_DEVICE_TRAIN_BATCH_SIZE}bs_perGPU-1024rel-2arity-0p1restrict-0p45reverse-pretrain-acc1"

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
  data.completion_only_loss="${COMPLETION_ONLY_LOSS}" \
  data.assistant_only_loss="${ASSISTANT_ONLY_LOSS}" \
  train.num_train_epochs="${NUM_TRAIN_EPOCHS}" \
  train.max_steps="${MAX_STEPS}" \
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
