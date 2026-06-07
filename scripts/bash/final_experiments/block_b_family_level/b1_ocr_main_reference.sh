#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

exec bash scripts/bash/final_experiments/block_a_ocr_vertical/a0_ocr_canonical_plain_k6_wd1.sh
