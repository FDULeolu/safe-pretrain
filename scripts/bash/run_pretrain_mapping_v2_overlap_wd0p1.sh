#!/usr/bin/env bash
set -euo pipefail

# Run the formal 0.3B-token, 1-epoch pretraining job on the mapping-v2 overlap
# dataset. Parameters match scripts/bash/run_pretrain.sh except for the dataset,
# experiment name, and weight decay.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/pretrain_a6000_smollm2_135m.yaml"

# Dataset knobs.
WORLD_NAME="synthetic_world_1024effects_512causes_0.1restricted_2arity_4x_overlap_dic-words"
RENDER_NAME="mapping_v2_0p3b_1024rel_512cause_4x_overlap_composable_v1_v3_random_swap"
DATASET_ROOT="${ROOT_DIR}/data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}"
BLOCK_SIZE=512
TOKENIZED_PATH="${DATASET_ROOT}/tokenized/bs${BLOCK_SIZE}"

# Hardware.
VISIBLE_DEVICES="4,5,6,7"
MIXED_PRECISION="bf16"

# Batch sizing.
# Global tokens/update = num_gpus * per_device_batch_size * grad_accum * block_size.
# With 4 GPUs, batch 64, grad_accum 1, block 512: 131,072 tokens/update.
PER_DEVICE_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1

# Optimizer/schedule.
NUM_TRAIN_EPOCHS=1
MAX_TRAIN_STEPS=null
LEARNING_RATE="3.0e-4"
WEIGHT_DECAY="1.0"
WARMUP_RATIO="0.03"
SCHEDULER="cosine"
MAX_GRAD_NORM="1.0"

# Runtime cadence.
LOG_EVERY_STEPS=10
EVAL_EVERY_STEPS=50
MAX_EVAL_BATCHES=20
SAVE_EVERY_STEPS=250
KEEP_LAST=3

# Dataloader/profiler.
DATALOADER_NUM_WORKERS=16
PREFETCH_FACTOR=4
PROFILER_ENABLED="true"
PROFILER_SYNC_CUDA="false"

# Run identity.
EXPERIMENT_NAME="smollm2-135m-scratch-0p3b-1epoch-bs${BLOCK_SIZE}-wd1-1024rel-512cause-4xoverlap-2arity-0p1restrict-mapping-v2-composable-v1-v3-random-swap"
WANDB_ENABLED="true"

# Keep Hugging Face cache local to this repo.
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

if [[ ! -d "${TOKENIZED_PATH}" ]]; then
  echo "Missing tokenized dataset: ${TOKENIZED_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TOKENIZED_PATH}/dataset_dict.json" || ! -f "${TOKENIZED_PATH}/metadata.json" ]]; then
  echo "Incomplete tokenized dataset: ${TOKENIZED_PATH}" >&2
  exit 1
fi

if [[ "${VISIBLE_DEVICES}" == "all" || "${VISIBLE_DEVICES}" == "null" || -z "${VISIBLE_DEVICES}" ]]; then
  NUM_GPUS=4
else
  IFS=',' read -r -a GPU_IDS <<< "${VISIBLE_DEVICES}"
  NUM_GPUS="${#GPU_IDS[@]}"
fi

GLOBAL_TOKENS_PER_STEP=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))

echo "Running formal pretrain:"
echo "  experiment: ${EXPERIMENT_NAME}"
echo "  tokenized path: ${TOKENIZED_PATH}"
echo "  visible devices: ${VISIBLE_DEVICES}"
echo "  num GPUs: ${NUM_GPUS}"
echo "  mixed precision: ${MIXED_PRECISION}"
echo "  per-device batch size: ${PER_DEVICE_BATCH_SIZE}"
echo "  gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  approx global tokens/update: ${GLOBAL_TOKENS_PER_STEP}"
echo "  epochs: ${NUM_TRAIN_EPOCHS}"
echo "  weight decay: ${WEIGHT_DECAY}"
echo "  wandb enabled: ${WANDB_ENABLED}"

conda run --no-capture-output -n safe-pretrain \
  python scripts/python/launch_pretrain.py \
    --config "${CONFIG}" \
    "project.experiment_name=${EXPERIMENT_NAME}" \
    "runtime.visible_devices=${VISIBLE_DEVICES}" \
    "runtime.mixed_precision=${MIXED_PRECISION}" \
    "data.tokenized.path=${TOKENIZED_PATH}" \
    "dataloader.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
    "dataloader.num_workers=${DATALOADER_NUM_WORKERS}" \
    "dataloader.prefetch_factor=${PREFETCH_FACTOR}" \
    "train.num_train_epochs=${NUM_TRAIN_EPOCHS}" \
    "train.max_train_steps=${MAX_TRAIN_STEPS}" \
    "train.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
    "train.learning_rate=${LEARNING_RATE}" \
    "train.weight_decay=${WEIGHT_DECAY}" \
    "train.warmup_ratio=${WARMUP_RATIO}" \
    "train.scheduler=${SCHEDULER}" \
    "train.max_grad_norm=${MAX_GRAD_NORM}" \
    "train.log_every_steps=${LOG_EVERY_STEPS}" \
    "train.eval_every_steps=${EVAL_EVERY_STEPS}" \
    "train.max_eval_batches=${MAX_EVAL_BATCHES}" \
    "checkpoint.save_every_steps=${SAVE_EVERY_STEPS}" \
    "checkpoint.keep_last=${KEEP_LAST}" \
    "wandb.enabled=${WANDB_ENABLED}" \
    "profiler.enabled=${PROFILER_ENABLED}" \
    "profiler.synchronize_cuda=${PROFILER_SYNC_CUDA}"
