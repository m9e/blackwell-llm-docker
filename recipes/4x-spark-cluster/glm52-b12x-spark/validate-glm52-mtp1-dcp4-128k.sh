#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

# shellcheck source=glm52-mtp1-dcp4-128k.env
source ./glm52-mtp1-dcp4-128k.env

HEAD_CONTAINER=${HEAD_CONTAINER:-glm-dark-head}
WORKER_CONTAINER=${WORKER_CONTAINER:-glm-dark-worker}
WORKER_HOSTS_STR=${WORKER_HOSTS:-"192.168.100.2 192.168.100.3 192.168.100.4"}
SSH_KEY=${SSH_KEY:-/etc/kamiwaza/ssl/cluster.key}
BASE_URL=${BASE_URL:-http://192.168.100.1:18089/v1}
EXPECTED_KV_TOKENS=${EXPECTED_KV_TOKENS:-132096}
EXPECTED_CONCURRENCY_PREFIX=${EXPECTED_CONCURRENCY_PREFIX:-1.01x}
DECODE_MAX_TOKENS=${DECODE_MAX_TOKENS:-128}
MIN_DECODE_TPS=${MIN_DECODE_TPS:-8.0}
OBJECT_STORE=${OBJECT_STORE:-134217728}
OBJECT_SPILLING_DIR=${OBJECT_SPILLING_DIR:-/var/tmp/ray-spill}

IFS=' ' read -r -a WORKER_HOSTS <<< "${WORKER_HOSTS_STR}"

pass() { printf 'PASS %s\n' "$*"; }
warn() { printf 'WARN %s\n' "$*" >&2; }
fail() { printf 'FAIL %s\n' "$*" >&2; exit 1; }

fmt_int() {
  python3 - "$1" <<'PY'
import sys
print(f"{int(sys.argv[1]):,}")
PY
}

ssh_run() {
  local host=$1
  shift
  ssh -i "${SSH_KEY}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    "matt@${host}" "$@"
}

container_image() {
  docker inspect -f '{{.Config.Image}}' "$1"
}

remote_container_image() {
  local host=$1
  ssh_run "${host}" docker inspect -f '{{.Config.Image}}' "${WORKER_CONTAINER}"
}

require_launch_flag() {
  local pattern=$1
  grep -Fq -- "${pattern}" ./launch-ray.sh || fail "launch-ray.sh missing ${pattern}"
  pass "launch-ray.sh contains ${pattern}"
}

require_log_contains() {
  local label=$1
  local pattern=$2
  local logs=$3
  grep -Fq -- "${pattern}" <<< "${logs}" || fail "${label} missing log marker: ${pattern}"
  pass "${label} contains log marker: ${pattern}"
}

maybe_log_contains() {
  local label=$1
  local pattern=$2
  local logs=$3
  if grep -Fq -- "${pattern}" <<< "${logs}"; then
    pass "${label} contains log marker: ${pattern}"
  else
    warn "${label} missing optional log marker: ${pattern}"
  fi
}

require_process_arg() {
  local label=$1
  local pattern=$2
  local args=$3
  grep -Fq -- "${pattern}" <<< "${args}" || fail "${label} missing process arg fragment: ${pattern}"
  pass "${label} process args contain ${pattern}"
}

check_no_log_monitor() {
  local label=$1
  local command_prefix=$2
  if ${command_prefix} bash -lc "pgrep -af '[l]og_monitor' >/tmp/kz-log-monitor.$$ 2>/dev/null; rc=\$?; cat /tmp/kz-log-monitor.$$ 2>/dev/null; rm -f /tmp/kz-log-monitor.$$; exit \$rc"; then
    fail "${label} has Ray log_monitor process despite --include-log-monitor=false"
  fi
  pass "${label} has no Ray log_monitor process"
}

check_dashboard_helper() {
  local label=$1
  local command_prefix=$2
  if ${command_prefix} bash -lc "pgrep -af '[d]ashboard.py' >/tmp/kz-dashboard.$$ 2>/dev/null; rc=\$?; cat /tmp/kz-dashboard.$$ 2>/dev/null; rm -f /tmp/kz-dashboard.$$; exit \$rc"; then
    warn "${label} has Ray dashboard.py helper process; Ray may retain UsageStatsHead even with --include-dashboard=false"
  else
    pass "${label} has no Ray dashboard.py helper process"
  fi
}

check_raylet_object_store_hint() {
  local label=$1
  local command_prefix=$2
  local args
  args=$(${command_prefix} bash -lc "ps -eo args | grep '[r]aylet' | head -n 1" || true)
  if [[ -z "${args}" ]]; then
    warn "${label} raylet args not observable"
    return
  fi
  if [[ "${args}" == *object_store_memory* ]]; then
    [[ "${args}" == *"${OBJECT_STORE}"* ]] || fail "${label} raylet object store is not ${OBJECT_STORE}: ${args}"
    pass "${label} raylet object store arg includes ${OBJECT_STORE}"
  else
    warn "${label} raylet args do not expose object_store_memory"
  fi
}

check_spill_dir() {
  local label=$1
  local command_prefix=$2
  ${command_prefix} test -d "${OBJECT_SPILLING_DIR}" || fail "${label} missing object spilling dir ${OBJECT_SPILLING_DIR}"
  pass "${label} has object spilling dir ${OBJECT_SPILLING_DIR}"
}

remote_worker_exec() {
  local host=$1
  local remote_cmd=$2
  ssh -i "${SSH_KEY}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    "matt@${host}" "${remote_cmd}"
}

check_remote_no_log_monitor() {
  local host=$1
  local out
  out=$(remote_worker_exec "${host}" "docker exec ${WORKER_CONTAINER} bash -lc 'pgrep -af \"[l]og_monitor\" || true'")
  [[ -z "${out}" ]] || fail "worker ${host} has Ray log_monitor process despite --include-log-monitor=false: ${out}"
  pass "worker ${host} has no Ray log_monitor process"
}

check_remote_dashboard_helper() {
  local host=$1
  local out
  out=$(remote_worker_exec "${host}" "docker exec ${WORKER_CONTAINER} bash -lc 'pgrep -af \"[d]ashboard.py\" || true'")
  if [[ -n "${out}" ]]; then
    warn "worker ${host} has Ray dashboard.py helper process: ${out}"
  else
    pass "worker ${host} has no Ray dashboard.py helper process"
  fi
}

check_remote_raylet_object_store_hint() {
  local host=$1
  local args
  args=$(remote_worker_exec "${host}" "docker exec ${WORKER_CONTAINER} bash -lc 'ps -eo args | grep \"[r]aylet\" | head -n 1'" || true)
  [[ -n "${args}" ]] || fail "worker ${host} raylet args not observable"
  [[ "${args}" == *object_store_memory* ]] || fail "worker ${host} raylet args do not expose object_store_memory"
  [[ "${args}" == *"${OBJECT_STORE}"* ]] || fail "worker ${host} raylet object store is not ${OBJECT_STORE}: ${args}"
  pass "worker ${host} raylet object store arg includes ${OBJECT_STORE}"
}

check_remote_spill_dir() {
  local host=$1
  remote_worker_exec "${host}" "docker exec ${WORKER_CONTAINER} test -d ${OBJECT_SPILLING_DIR}" || fail "worker ${host} missing object spilling dir ${OBJECT_SPILLING_DIR}"
  pass "worker ${host} has object spilling dir ${OBJECT_SPILLING_DIR}"
}

head_image=$(container_image "${HEAD_CONTAINER}")
[[ "${head_image}" == "${IMAGE}" ]] || fail "head image ${head_image}, expected ${IMAGE}"
pass "head image is ${IMAGE}"

for host in "${WORKER_HOSTS[@]}"; do
  worker_image=$(remote_container_image "${host}")
  [[ "${worker_image}" == "${IMAGE}" ]] || fail "worker ${host} image ${worker_image}, expected ${IMAGE}"
  pass "worker ${host} image is ${IMAGE}"
done

require_launch_flag '--include-dashboard=false'
require_launch_flag '--disable-usage-stats'
require_launch_flag '--include-log-monitor=false'
require_launch_flag '134217728'
require_launch_flag '--object-store-memory'
require_launch_flag '/var/tmp/ray-spill'
require_launch_flag '--object-spilling-directory'

head_logs=$(docker logs "${HEAD_CONTAINER}" 2>&1)
if grep -Fq 'Usage stats collection is enabled' <<< "${head_logs}"; then
  fail 'Ray usage stats enabled message found in head logs'
fi
pass 'Ray usage stats enabled message absent from head logs'

check_no_log_monitor 'head' 'docker exec glm-dark-head'
check_dashboard_helper 'head' 'docker exec glm-dark-head'
check_raylet_object_store_hint 'head' 'docker exec glm-dark-head'
check_spill_dir 'head' 'docker exec glm-dark-head'

for host in "${WORKER_HOSTS[@]}"; do
  check_remote_no_log_monitor "${host}"
  check_remote_dashboard_helper "${host}"
  check_remote_raylet_object_store_hint "${host}"
  check_remote_spill_dir "${host}"
done

api_args=$(docker exec "${HEAD_CONTAINER}" bash -lc "ps -eo args | grep '[v]llm.entrypoints.openai.api_server'")
require_process_arg 'api server' "--served-model-name ${SERVED_MODEL_NAME}" "${api_args}"
require_process_arg 'api server' "--tensor-parallel-size ${TP_SIZE}" "${api_args}"
require_process_arg 'api server' "--decode-context-parallel-size ${DCP_SIZE}" "${api_args}"
require_process_arg 'api server' "--dcp-comm-backend ${DCP_COMM_BACKEND}" "${api_args}"
require_process_arg 'api server' "--max-model-len ${MAX_MODEL_LEN}" "${api_args}"
require_process_arg 'api server' "--max-num-seqs ${MAX_NUM_SEQS}" "${api_args}"
require_process_arg 'api server' "--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}" "${api_args}"
require_process_arg 'api server' "--max-cudagraph-capture-size ${MAX_CUDAGRAPH_CAPTURE_SIZE}" "${api_args}"
require_process_arg 'api server' "--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}" "${api_args}"
require_process_arg 'api server' "--kv-cache-dtype ${KV_CACHE_DTYPE}" "${api_args}"
require_process_arg 'api server' "--attention-backend ${ATTENTION_BACKEND}" "${api_args}"
require_process_arg 'api server' "--moe-backend ${MOE_BACKEND}" "${api_args}"
if [[ "${ENABLE_MTP}" == "1" ]]; then
  require_process_arg 'api server' '"method":"mtp"' "${api_args}"
  require_process_arg 'api server' "num_speculative_tokens" "${api_args}"
  require_process_arg 'api server' "${NUM_SPECULATIVE_TOKENS}" "${api_args}"
fi

head_ray_logs=$(docker exec "${HEAD_CONTAINER}" bash -lc "grep -R -F -e 'GPU KV cache size' -e 'Maximum concurrency' -e 'VLLM_KZ_TRIM_AFTER_LOAD completed' -e 'Application startup complete' /tmp/ray-vllm-head/session_latest/logs 2>/dev/null || true")
head_combined_logs="${head_logs}"$'
'"${head_ray_logs}"
EXPECTED_KV_TOKENS_DISPLAY=$(fmt_int "${EXPECTED_KV_TOKENS}")
MAX_MODEL_LEN_DISPLAY=$(fmt_int "${MAX_MODEL_LEN}")
maybe_log_contains 'head' "GPU KV cache size: ${EXPECTED_KV_TOKENS_DISPLAY} tokens" "${head_combined_logs}"
maybe_log_contains 'head' "Maximum concurrency for ${MAX_MODEL_LEN_DISPLAY} tokens per request: ${EXPECTED_CONCURRENCY_PREFIX}" "${head_combined_logs}"
maybe_log_contains 'head' 'Application startup complete.' "${head_combined_logs}"
require_log_contains 'head' 'VLLM_KZ_TRIM_AFTER_LOAD completed' "${head_combined_logs}"

for host in "${WORKER_HOSTS[@]}"; do
  worker_logs=$(ssh_run "${host}" docker logs "${WORKER_CONTAINER}" 2>&1 || true)
  worker_ray_logs=$(ssh -i "${SSH_KEY}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    "matt@${host}" \
    "docker exec ${WORKER_CONTAINER} bash -lc 'grep -R -F "VLLM_KZ_TRIM_AFTER_LOAD completed" /tmp /var/tmp 2>/dev/null | tail -n 20 || true'" || true)
  worker_combined_logs="${worker_logs}"$'
'"${worker_ray_logs}"
  require_log_contains "worker ${host}" 'VLLM_KZ_TRIM_AFTER_LOAD completed' "${worker_combined_logs}"
done

MODELS_JSON=$(curl -fsS "${BASE_URL}/models") \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
python3 - <<'PY'
import json
import os
import sys

data = json.loads(os.environ["MODELS_JSON"])
served = os.environ["SERVED_MODEL_NAME"]
expected_len = int(os.environ["MAX_MODEL_LEN"])
for model in data.get("data", []):
    if model.get("id") != served:
        continue
    actual_len = model.get("max_model_len")
    if actual_len != expected_len:
        print(f"FAIL /v1/models max_model_len={actual_len}, expected {expected_len}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS /v1/models exposes {served} max_model_len={actual_len}")
    raise SystemExit(0)
print(f"FAIL /v1/models missing {served}", file=sys.stderr)
raise SystemExit(1)
PY

BASE_URL="${BASE_URL}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS}" \
MIN_DECODE_TPS="${MIN_DECODE_TPS}" \
python3 - <<'PY'
import json
import os
import sys
import time
import urllib.request

base_url = os.environ["BASE_URL"].rstrip("/")
model = os.environ["SERVED_MODEL_NAME"]
max_tokens = int(os.environ["DECODE_MAX_TOKENS"])
min_tps = float(os.environ["MIN_DECODE_TPS"])
payload = {
    "model": model,
    "prompt": "Write a concise Python function that parses a CSV string and returns a list of dictionaries. Include only code and brief comments.",
    "max_tokens": max_tokens,
    "temperature": 0.0,
}
body = json.dumps(payload).encode()
req = urllib.request.Request(
    f"{base_url}/completions",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
start = time.perf_counter()
with urllib.request.urlopen(req, timeout=300) as response:
    data = json.loads(response.read().decode())
elapsed = time.perf_counter() - start
usage = data.get("usage") or {}
completion_tokens = usage.get("completion_tokens")
if not completion_tokens:
    text = "".join(choice.get("text", "") for choice in data.get("choices", []))
    completion_tokens = max(1, len(text.split()))
tps = completion_tokens / elapsed
if tps < min_tps:
    print(
        f"FAIL decode throughput {tps:.3f} tok/s below minimum {min_tps:.3f} tok/s "
        f"({completion_tokens} tokens in {elapsed:.3f}s)",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"PASS decode throughput {tps:.3f} tok/s ({completion_tokens} tokens in {elapsed:.3f}s)")
PY
