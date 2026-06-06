#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

WAIT_SESSION="safe_pretrain_mapping_v2_overlap"
PRETRAIN_SCRIPT="${ROOT_DIR}/scripts/bash/run_pretrain_mapping_v2_overlap_wd0p1.sh"
WORLD_NAME="synthetic_world_1024effects_512causes_0.1restricted_2arity_4x_overlap_dic-words"
RENDER_NAME="mapping_v2_0p3b_1024rel_512cause_4x_overlap_composable_v1_v3_random_swap"
TOKENIZED_PATH="${ROOT_DIR}/data/worlds/${WORLD_NAME}/pretrain/${RENDER_NAME}/tokenized/bs512"
LOG_DIR="${ROOT_DIR}/logs/pretrain_data"
LOG_FILE="${LOG_DIR}/wait_mapping_v2_overlap_then_pretrain_wd0p1.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date -Is)] Waiting for tmux session ${WAIT_SESSION}"
while tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -Fxq "${WAIT_SESSION}"; do
  echo "[$(date -Is)] ${WAIT_SESSION} still running"
  sleep 60
done

echo "[$(date -Is)] ${WAIT_SESSION} finished"
if [[ ! -d "${TOKENIZED_PATH}" ]]; then
  echo "Missing tokenized dataset: ${TOKENIZED_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TOKENIZED_PATH}/dataset_dict.json" || ! -f "${TOKENIZED_PATH}/metadata.json" ]]; then
  echo "Incomplete tokenized dataset: ${TOKENIZED_PATH}" >&2
  exit 1
fi

echo "[$(date -Is)] Starting wd0p1 pretrain"
bash "${PRETRAIN_SCRIPT}"
echo "[$(date -Is)] Finished wd0p1 pretrain"
