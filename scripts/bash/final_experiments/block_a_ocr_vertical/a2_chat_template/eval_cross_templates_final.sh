#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

if [[ "${RUN_EVAL_SFT:-true}" != "true" ]]; then
  echo "RUN_EVAL_SFT=false; skipping cross-template eval"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${EVAL_DEVICE:-auto}"
DTYPE="${EVAL_DTYPE:-auto}"
BATCH_SIZE="${SFT_EVAL_BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${SFT_EVAL_MAX_NEW_TOKENS:-32}"
MAX_EXAMPLES="${SFT_EVAL_MAX_EXAMPLES:-null}"

PLAIN_SFT_DIR="data/experiments/blockA_a0_ocr_w1024_c2048_a1_r10_plain_fi6_wd1/sft_plain"
CHATML_SFT_DIR="data/experiments/blockA_a2_ocr_w1024_a1_chatml_fi6/sft_chatml"
PLAIN_MODEL="outputs/sft-blockA_a0_ocr_plain_fi6_wd1/final_model"
CHATML_MODEL="outputs/sft-blockA_a2_ocr_chatml_fi6/final_model"
OUTPUT_ROOT="outputs/eval-cross-template/blockA_a2_ocr"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required cross-template eval dependency: ${path}" >&2
    echo "Run the canonical plain and chatml SFT experiments first." >&2
    exit 2
  fi
}

require_path "${PLAIN_SFT_DIR}/chat_template.jinja"
require_path "${CHATML_SFT_DIR}/chat_template.jinja"
require_path "${PLAIN_MODEL}"
require_path "${CHATML_MODEL}"

"${PYTHON_BIN}" scripts/python/eval_sft_qa.py \
  --model "${PLAIN_MODEL}" \
  --sft-dir "${PLAIN_SFT_DIR}" \
  --chat-template-path "${CHATML_SFT_DIR}/chat_template.jinja" \
  --output-dir "${OUTPUT_ROOT}/plain_train_chatml_eval_final" \
  --max-examples "${MAX_EXAMPLES}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

"${PYTHON_BIN}" scripts/python/eval_sft_qa.py \
  --model "${CHATML_MODEL}" \
  --sft-dir "${CHATML_SFT_DIR}" \
  --chat-template-path "${PLAIN_SFT_DIR}/chat_template.jinja" \
  --output-dir "${OUTPUT_ROOT}/chatml_train_plain_eval_final" \
  --max-examples "${MAX_EXAMPLES}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
