#!/usr/bin/env bash
set -euo pipefail

# Run a short smoke pretraining job from an already-tokenized dataset.
# Edit the variables below, then run:
#   bash scripts/bash/run_smoke_pretrain.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Main knobs.
CONFIG="configs/pretrain_a6000_smollm2_135m.yaml"
EXPERIMENT_NAME="smoke-bs512"
TOKENIZED_PATH="data/tokenized/smoke_bs512"
VISIBLE_DEVICES="0,1,2,3"
WANDB_ENABLED="true"

# Training knobs.
MAX_TRAIN_STEPS=20
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE="3.0e-4"

# Smoke-run cadence.
LOG_EVERY_STEPS=1
EVAL_EVERY_STEPS=10
SAVE_EVERY_STEPS=10

# Less common knobs.
MIXED_PRECISION="auto"
DATALOADER_NUM_WORKERS=2
MAX_EVAL_BATCHES=10
KEEP_LAST=2
PROFILER_SYNC_CUDA="false"

# Use the local cache populated during smoke tokenization.
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

if [[ ! -d "${TOKENIZED_PATH}" ]]; then
  echo "Missing tokenized dataset: ${TOKENIZED_PATH}" >&2
  echo "Expected an already-tokenized dataset, e.g. data/tokenized/smoke_bs512." >&2
  exit 1
fi

echo "Running smoke pretrain:"
echo "  experiment: ${EXPERIMENT_NAME}"
echo "  tokenized path: ${TOKENIZED_PATH}"
echo "  visible devices: ${VISIBLE_DEVICES}"
echo "  max train steps: ${MAX_TRAIN_STEPS}"
echo "  wandb enabled: ${WANDB_ENABLED}"

python scripts/python/launch_pretrain.py \
  --config "${CONFIG}" \
  "project.experiment_name=${EXPERIMENT_NAME}" \
  "runtime.visible_devices=${VISIBLE_DEVICES}" \
  "runtime.mixed_precision=${MIXED_PRECISION}" \
  "data.tokenized.path=${TOKENIZED_PATH}" \
  "dataloader.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
  "dataloader.num_workers=${DATALOADER_NUM_WORKERS}" \
  "train.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  "train.max_train_steps=${MAX_TRAIN_STEPS}" \
  "train.learning_rate=${LEARNING_RATE}" \
  "train.log_every_steps=${LOG_EVERY_STEPS}" \
  "train.eval_every_steps=${EVAL_EVERY_STEPS}" \
  "train.max_eval_batches=${MAX_EVAL_BATCHES}" \
  "checkpoint.save_every_steps=${SAVE_EVERY_STEPS}" \
  "checkpoint.keep_last=${KEEP_LAST}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "profiler.synchronize_cuda=${PROFILER_SYNC_CUDA}"
