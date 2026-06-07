#!/usr/bin/env bash
set -euo pipefail

# Launch the final experiment matrix one job at a time by default:
#   slot 0 -> GPUs 0,1,2,3 and port 29510
#
# Set FINAL_EXPERIMENT_CONCURRENCY=2 to use two concurrent slots:
#   slot 1 -> GPUs 4,5,6,7 and port 29520
#
# The schedule is dependency-aware. It first runs the canonical OCR experiment
# as a barrier, then keeps up to FINAL_EXPERIMENT_CONCURRENCY downstream jobs
# active. Reference aliases that would write the same outputs are intentionally
# not launched.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SLOT0_VISIBLE_DEVICES="${SLOT0_VISIBLE_DEVICES:-0,1,2,3}"
SLOT1_VISIBLE_DEVICES="${SLOT1_VISIBLE_DEVICES:-4,5,6,7}"
SLOT0_MAIN_PROCESS_PORT="${SLOT0_MAIN_PROCESS_PORT:-29510}"
SLOT1_MAIN_PROCESS_PORT="${SLOT1_MAIN_PROCESS_PORT:-29520}"
CHECK_PORTS="${CHECK_PORTS:-true}"
SCHEDULER_DRY_RUN="${SCHEDULER_DRY_RUN:-false}"
FINAL_EXPERIMENT_CONCURRENCY="${FINAL_EXPERIMENT_CONCURRENCY:-1}"
LOG_ROOT="${LOG_ROOT:-logs/final_experiments/$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

fail() {
  echo "run_all_final_experiments failed: $*" >&2
  exit 2
}

log() {
  echo "[final-runner] $*"
}

if [[ "${RUN_DATA:-true}" == "true" && "${OVERWRITE_DATA:-false}" == "true" ]]; then
  fail "refusing OVERWRITE_DATA=true in the main scheduler; rebuild individual experiments explicitly"
fi

case "${FINAL_EXPERIMENT_CONCURRENCY}" in
  1 | 2) ;;
  *) fail "FINAL_EXPERIMENT_CONCURRENCY must be 1 or 2, got ${FINAL_EXPERIMENT_CONCURRENCY}" ;;
esac

if [[ "${FINAL_EXPERIMENT_CONCURRENCY}" == "2" && "${SLOT0_MAIN_PROCESS_PORT}" == "${SLOT1_MAIN_PROCESS_PORT}" ]]; then
  fail "slot ports must differ: both are ${SLOT0_MAIN_PROCESS_PORT}"
fi

if ! command -v setsid >/dev/null 2>&1; then
  fail "setsid is required so the scheduler can clean up full experiment process groups"
fi

PORT_CHECK_PYTHON="${PYTHON_BIN}"
if ! command -v "${PORT_CHECK_PYTHON}" >/dev/null 2>&1; then
  PORT_CHECK_PYTHON="python"
fi

check_port_available() {
  local port="$1"
  if [[ "${CHECK_PORTS}" != "true" || "${SCHEDULER_DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "${PORT_CHECK_PYTHON}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
except PermissionError as exc:
    print(f"port check skipped for {port}: socket check is not permitted in this environment ({exc})", file=sys.stderr)
except OSError as exc:
    raise SystemExit(f"port {port} is not available: {exc}") from exc
PY
}

slot_devices() {
  case "$1" in
    0) echo "${SLOT0_VISIBLE_DEVICES}" ;;
    1) echo "${SLOT1_VISIBLE_DEVICES}" ;;
    *) fail "unknown slot: $1" ;;
  esac
}

slot_port() {
  case "$1" in
    0) echo "${SLOT0_MAIN_PROCESS_PORT}" ;;
    1) echo "${SLOT1_MAIN_PROCESS_PORT}" ;;
    *) fail "unknown slot: $1" ;;
  esac
}

sanitize_name() {
  echo "$1" | tr '/ :' '___'
}

ACTIVE_PIDS=()
SLOT_PIDS=()
declare -A PID_TO_NAME=()
declare -A PID_TO_SLOT=()
JOB_NAMES=()
JOB_SCRIPTS=()

cleanup_children() {
  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    echo "Stopping active jobs: ${ACTIVE_PIDS[*]}" >&2
    local pid
    for pid in "${ACTIVE_PIDS[@]}"; do
      kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    done
  fi
}

trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM

launch_job() {
  local slot="$1"
  local name="$2"
  local script="$3"
  local devices
  local port
  local log_file

  [[ -f "${script}" ]] || fail "missing experiment script: ${script}"
  devices="$(slot_devices "${slot}")"
  port="$(slot_port "${slot}")"
  check_port_available "${port}"
  log_file="${LOG_ROOT}/$(printf 'slot%s_%s.log' "${slot}" "$(sanitize_name "${name}")")"

  log "slot ${slot}: ${name}"
  log "  script=${script}"
  log "  GPUs=${devices} port=${port}"
  log "  log=${log_file}"

  if [[ "${SCHEDULER_DRY_RUN}" == "true" ]]; then
    LAST_PID=""
    return 0
  fi

  mkdir -p "${LOG_ROOT}"
  setsid env \
    JOB_NAME="${name}" \
    JOB_SCRIPT="${script}" \
    JOB_VISIBLE_DEVICES="${devices}" \
    JOB_MAIN_PROCESS_PORT="${port}" \
    JOB_PYTHON_BIN="${PYTHON_BIN}" \
    bash -c '
      set -euo pipefail
      echo "[job] name=${JOB_NAME}"
      echo "[job] script=${JOB_SCRIPT}"
      echo "[job] visible_devices=${JOB_VISIBLE_DEVICES}"
      echo "[job] main_process_port=${JOB_MAIN_PROCESS_PORT}"
      echo "[job] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      VISIBLE_DEVICES="${JOB_VISIBLE_DEVICES}" \
      MAIN_PROCESS_PORT="${JOB_MAIN_PROCESS_PORT}" \
      PYTHON_BIN="${JOB_PYTHON_BIN}" \
        bash "${JOB_SCRIPT}"
      echo "[job] finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ' >"${log_file}" 2>&1 &
  LAST_PID="$!"
}

enqueue_job() {
  local name="$1"
  local script="$2"
  JOB_NAMES+=("${name}")
  JOB_SCRIPTS+=("${script}")
}

active_count() {
  local count=0
  local pid
  for pid in "${SLOT_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

refresh_active_pids() {
  ACTIVE_PIDS=()
  local pid
  for pid in "${SLOT_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      ACTIVE_PIDS+=("${pid}")
    fi
  done
}

free_slot() {
  local slot
  for ((slot = 0; slot < FINAL_EXPERIMENT_CONCURRENCY; slot++)); do
    if [[ -z "${SLOT_PIDS[slot]:-}" ]]; then
      echo "${slot}"
      return 0
    fi
  done
  return 1
}

launch_queued_job() {
  local slot="$1"
  local index="$2"
  local name="${JOB_NAMES[index]}"
  local script="${JOB_SCRIPTS[index]}"
  local pid=""

  launch_job "${slot}" "${name}" "${script}"
  pid="${LAST_PID}"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  SLOT_PIDS[slot]="${pid}"
  PID_TO_NAME["${pid}"]="${name}"
  PID_TO_SLOT["${pid}"]="${slot}"
  refresh_active_pids
}

run_job_queue() {
  local total="${#JOB_NAMES[@]}"
  local next=0
  local slot=""
  local finished_pid=""
  local finished_slot=""
  local finished_name=""
  local status=0

  if [[ "${SCHEDULER_DRY_RUN}" == "true" ]]; then
    for ((next = 0; next < total; next++)); do
      slot=$((next % FINAL_EXPERIMENT_CONCURRENCY))
      launch_queued_job "${slot}" "${next}"
    done
    return 0
  fi

  while (( next < total || $(active_count) > 0 )); do
    while (( next < total )) && slot="$(free_slot)"; do
      launch_queued_job "${slot}" "${next}"
      next=$((next + 1))
    done

    if (( $(active_count) == 0 )); then
      continue
    fi

    finished_pid=""
    status=0
    wait -n -p finished_pid "${ACTIVE_PIDS[@]}" || status=$?
    if [[ -z "${finished_pid}" ]]; then
      fail "wait returned without a finished pid; inspect ${LOG_ROOT}"
    fi

    finished_name="${PID_TO_NAME[${finished_pid}]:-unknown}"
    finished_slot="${PID_TO_SLOT[${finished_pid}]:-}"
    if [[ -n "${finished_slot}" ]]; then
      SLOT_PIDS[finished_slot]=""
    fi
    unset "PID_TO_NAME[${finished_pid}]"
    unset "PID_TO_SLOT[${finished_pid}]"
    refresh_active_pids

    if [[ "${status}" -ne 0 ]]; then
      kill -- "-${finished_pid}" 2>/dev/null || kill "${finished_pid}" 2>/dev/null || true
      cleanup_children
      fail "${finished_name} failed with status ${status}; inspect ${LOG_ROOT}"
    fi
    log "completed: ${finished_name}"
  done
}

clear_job_queue() {
  JOB_NAMES=()
  JOB_SCRIPTS=()
}

log "log_root=${LOG_ROOT}"
log "concurrency=${FINAL_EXPERIMENT_CONCURRENCY}"
log "slot0 GPUs=${SLOT0_VISIBLE_DEVICES} port=${SLOT0_MAIN_PROCESS_PORT}"
if [[ "${FINAL_EXPERIMENT_CONCURRENCY}" == "2" ]]; then
  log "slot1 GPUs=${SLOT1_VISIBLE_DEVICES} port=${SLOT1_MAIN_PROCESS_PORT}"
fi
if [[ "${SCHEDULER_DRY_RUN}" == "true" ]]; then
  log "dry run only; no jobs will be launched"
fi

enqueue_job \
  "blockA_a0_canonical_ocr_plain_fi6_wd1" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a0_ocr_canonical_plain_k6_wd1.sh"

log "=== stage 1: canonical OCR barrier ==="
run_job_queue
clear_job_queue

enqueue_job \
  "blockA_a1_ocr_plain_fi1" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a1_fi_k_sweep/fi_k1.sh"

enqueue_job \
  "blockA_a1_ocr_plain_fi2" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a1_fi_k_sweep/fi_k2.sh"

enqueue_job \
  "blockA_a1_ocr_plain_fi4" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a1_fi_k_sweep/fi_k4.sh"

enqueue_job \
  "blockA_a1_ocr_plain_fi8" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a1_fi_k_sweep/fi_k8.sh"

enqueue_job \
  "blockA_a2_ocr_chatml_fi6" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a2_chat_template/chatml_train_k6.sh"

enqueue_job \
  "blockA_a3_ocr_wd0p1" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a3_weight_decay/wd0p1.sh"

enqueue_job \
  "blockA_a3_ocr_wd2" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a3_weight_decay/wd2.sh"

enqueue_job \
  "blockA_a4_ocr_rel512" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a4_relation_count/rel512.sh"

enqueue_job \
  "blockA_a4_ocr_rel2048" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a4_relation_count/rel2048.sh"

enqueue_job \
  "blockA_a5_ocr_arity2_overlap" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a5_arity_overlap/arity2_overlap.sh"

enqueue_job \
  "blockB_b0_vanilla" \
  "scripts/bash/final_experiments/block_b_family_level/b0_vanilla_control.sh"

enqueue_job \
  "blockB_b2_ocr_linear" \
  "scripts/bash/final_experiments/block_b_family_level/b2_ocr_linear.sh"

enqueue_job \
  "blockB_b3_prevention" \
  "scripts/bash/final_experiments/block_b_family_level/b3_prevention.sh"

enqueue_job \
  "blockB_b4_mirror" \
  "scripts/bash/final_experiments/block_b_family_level/b4_mirror.sh"

log "=== stage 2: downstream experiment queue ==="
run_job_queue
clear_job_queue

enqueue_job \
  "blockA_a2_cross_template_eval" \
  "scripts/bash/final_experiments/block_a_ocr_vertical/a2_chat_template/eval_cross_templates_final.sh"

log "=== stage 3: cross-template eval barrier ==="
run_job_queue

log "all scheduled final experiments completed"
log "reference aliases intentionally covered by canonical: a1 fi_k6, a3 wd1, a4 rel1024, a5 arity1, blockB b1 OCR"
