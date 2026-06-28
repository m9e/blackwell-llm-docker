#!/usr/bin/env bash
set -euo pipefail

# Capacity-discovery harness for GLM-5.2 on the 4x Spark B12X stack.
# This intentionally does not run throughput prompts. It launches vLLM,
# waits for readiness/failure, and archives KV accounting diagnostics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-glm-darkdevotion-b12x:20260625-arm64-mtp1-trim}"
HEAD_IP="${HEAD_IP:-192.168.100.1}"
WORKER_IPS="${WORKER_IPS:-192.168.100.2 192.168.100.3 192.168.100.4}"
SSH_KEY="${SSH_KEY:-/etc/kamiwaza/ssl/cluster.key}"
HEAD_NAME="${HEAD_NAME:-glm-dark-head}"
WORKER_NAME="${WORKER_NAME:-glm-dark-worker}"
PORT="${PORT:-18089}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.918}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:--1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-4}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
RUN_DIR="${RUN_DIR:-/tmp/glm52-kv-capacity-$(date +%Y%m%d-%H%M%S)}"
CASES="${CASES:-A B C D}"

NO_MTP_MODEL="${NO_MTP_MODEL:-/var/tmp/models/Mapika/GLM-5.2-NVFP4}"
MTP_MODEL="${MTP_MODEL:-/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid}"

mkdir -p "${RUN_DIR}"

ssh_base=(
  ssh
  -i "${SSH_KEY}"
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
  -o BatchMode=yes
  -o ConnectTimeout=10
)

scp_base=(
  scp
  -i "${SSH_KEY}"
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
  -o BatchMode=yes
  -o ConnectTimeout=10
)

case_config() {
  case "$1" in
    A) echo "glm52-cap-A-tp4-dcp1-nomtp|${NO_MTP_MODEL}|1|0" ;;
    B) echo "glm52-cap-B-tp4-dcp4-nomtp|${NO_MTP_MODEL}|4|0" ;;
    C) echo "glm52-cap-C-tp4-dcp1-mtp3|${MTP_MODEL}|1|1" ;;
    D) echo "glm52-cap-D-tp4-dcp4-mtp3|${MTP_MODEL}|4|1" ;;
    *) echo "unknown case: $1" >&2; return 1 ;;
  esac
}

archive_logs() {
  local name="$1"
  local api_log="/tmp/${name}.log"

  docker cp "${HEAD_NAME}:${api_log}" "${RUN_DIR}/${name}.api.log" >/dev/null 2>&1 || true
  if [[ -f "${RUN_DIR}/${name}.api.log" ]]; then
    rg 'KZ_KV_DIAG|GPU KV cache size|Maximum concurrency|Available KV cache memory|Memory profiling|model weights|non_torch|torch_peak|No available memory|Cannot auto-fit|Traceback|ERROR' \
      "${RUN_DIR}/${name}.api.log" > "${RUN_DIR}/${name}.diag.txt" || true
  fi

  docker exec "${HEAD_NAME}" bash -lc \
    "tar -C /tmp/ray-vllm-head/session_latest/logs -czf /tmp/${name}.ray-head.tgz . 2>/dev/null || true" \
    >/dev/null 2>&1 || true
  docker cp "${HEAD_NAME}:/tmp/${name}.ray-head.tgz" "${RUN_DIR}/${name}.ray-head.tgz" >/dev/null 2>&1 || true

  for ip in ${WORKER_IPS}; do
    local safe_ip="${ip//./-}"
    "${ssh_base[@]}" "${ip}" "docker exec '${WORKER_NAME}' bash -lc 'tar -C /tmp/ray-vllm-worker/session_latest/logs -czf /tmp/${name}.ray-worker-${safe_ip}.tgz . 2>/dev/null || true' >/dev/null 2>&1 || true
docker cp '${WORKER_NAME}:/tmp/${name}.ray-worker-${safe_ip}.tgz' '/tmp/${name}.ray-worker-${safe_ip}.tgz' >/dev/null 2>&1 || true" || true
    "${scp_base[@]}" "${ip}:/tmp/${name}.ray-worker-${safe_ip}.tgz" "${RUN_DIR}/${name}.ray-worker-${safe_ip}.tgz" >/dev/null 2>&1 || true
    "${ssh_base[@]}" "${ip}" "rm -f '/tmp/${name}.ray-worker-${safe_ip}.tgz'" >/dev/null 2>&1 || true
  done

  {
    echo "== head ${HEAD_IP} =="
    docker exec "${HEAD_NAME}" bash -lc \
      "grep -R 'KZ_KV_DIAG_\\(WORKER_ENV\\|MEMORY\\)' -n /tmp/ray-vllm-head/session_latest/logs 2>/dev/null || true" || true
    for ip in ${WORKER_IPS}; do
      echo "== worker ${ip} =="
      "${ssh_base[@]}" "${ip}" \
        "docker exec '${WORKER_NAME}' bash -lc \"grep -R 'KZ_KV_DIAG_\\\\(WORKER_ENV\\\\|MEMORY\\\\)' -n /tmp/ray-vllm-worker/session_latest/logs 2>/dev/null || true\"" || true
    done
  } > "${RUN_DIR}/${name}.worker-envmem.txt"
}

wait_for_case() {
  local name="$1"
  local rc=1
  for poll in $(seq 1 180); do
    if curl -fsS "http://${HEAD_IP}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] ${name} READY"
      rc=0
      break
    fi
    if docker exec "${HEAD_NAME}" bash -lc "grep -qE 'Engine core initialization failed|Traceback|Cannot auto-fit|No available memory|ValueError: Free memory' /tmp/${name}.log 2>/dev/null"; then
      echo "[$(date +%H:%M:%S)] ${name} FAILED"
      rc=1
      break
    fi
    if (( poll % 10 == 1 )); then
      echo "[$(date +%H:%M:%S)] ${name} waiting poll=${poll}"
      docker exec "${HEAD_NAME}" bash -lc \
        "grep -R -E 'Loading safetensors|Using max model len|KZ_KV_DIAG_MEMORY|KZ_KV_DIAG_AVAILABLE|GPU KV cache size|Cannot auto-fit|No available memory|Traceback|ERROR' /tmp/${name}.log /tmp/ray-vllm-head/session_latest/logs 2>/dev/null | tail -n 30 || true" || true
    fi
    sleep 30
  done
  return "${rc}"
}

run_case() {
  local case_id="$1"
  local name model_dir dcp enable_mtp
  IFS='|' read -r name model_dir dcp enable_mtp <<< "$(case_config "${case_id}")"

  echo "=== ${name}: model=${model_dir} dcp=${dcp} mtp=${enable_mtp} util=${GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN} ==="

  MODEL_DIR="${model_dir}" \
  IMAGE="${IMAGE}" \
  KZ_KV_DIAG=1 \
  VLLM_USE_B12X_MOE=0 \
  VLLM_USE_B12X_FP8_GEMM=0 \
  B12X_MOE_FORCE_A16=0 \
  B12X_W4A16_TC_DECODE=0 \
    "${SCRIPT_DIR}/launch-ray.sh"

  KZ_KV_DIAG=1 \
    "${SCRIPT_DIR}/patch-vllm-diagnostics.sh"

  MODEL_DIR=/models \
  SERVED_MODEL_NAME="${name}" \
  PROFILE=custom \
  TP_SIZE=4 \
  DCP_SIZE="${dcp}" \
  PP_SIZE=1 \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  KV_CACHE_DTYPE=fp8 \
  ENABLE_MTP="${enable_mtp}" \
  NUM_SPECULATIVE_TOKENS=3 \
  ATTENTION_BACKEND=B12X_MLA_SPARSE \
  MOE_BACKEND=flashinfer_cutlass \
  VLLM_USE_B12X_MOE=0 \
  VLLM_USE_B12X_FP8_GEMM=0 \
  B12X_MOE_FORCE_A16=0 \
  B12X_W4A16_TC_DECODE=0 \
  ENFORCE_EAGER="${ENFORCE_EAGER}" \
  MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES}" \
  KZ_KV_DIAG=1 \
  PORT="${PORT}" \
  LOG_FILE="/tmp/${name}.log" \
    "${SCRIPT_DIR}/serve.sh"

  local rc=0
  wait_for_case "${name}" || rc=$?
  archive_logs "${name}"
  echo "${name} rc=${rc}" | tee -a "${RUN_DIR}/summary.txt"
  return "${rc}"
}

main() {
  local overall=0
  echo "RUN_DIR=${RUN_DIR}" | tee "${RUN_DIR}/run.env"
  echo "CASES=${CASES}" | tee -a "${RUN_DIR}/run.env"
  echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" | tee -a "${RUN_DIR}/run.env"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}" | tee -a "${RUN_DIR}/run.env"
  echo "KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES}" | tee -a "${RUN_DIR}/run.env"
  echo "ENFORCE_EAGER=${ENFORCE_EAGER}" | tee -a "${RUN_DIR}/run.env"

  for case_id in ${CASES}; do
    run_case "${case_id}" || overall=$?
  done

  echo "RUN_DIR=${RUN_DIR}"
  cat "${RUN_DIR}/summary.txt" 2>/dev/null || true
  return "${overall}"
}

main "$@"
