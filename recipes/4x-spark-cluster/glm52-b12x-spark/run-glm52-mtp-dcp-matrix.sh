#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_ENV="${BASE_ENV:-${SCRIPT_DIR}/glm52-mtp3-dcp4-128k-mtpgroups1.env}"
LAUNCHER="${LAUNCHER:-${SCRIPT_DIR}/launch-glm52-mtp3-dcp4-128k.sh}"
HEAD_NAME="${HEAD_NAME:-glm-dark-head}"
HEAD_IP="${HEAD_IP:-192.168.100.1}"
PORT="${PORT:-18089}"
SSH_KEY="${SSH_KEY:-/etc/kamiwaza/ssl/cluster.key}"
WORKER_IPS=(192.168.100.2 192.168.100.3 192.168.100.4)

DCP_LIST=(${DCP_LIST:-1 2 4})
MTP_LIST=(${MTP_LIST:-0 1 2 3})
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1900000000}"
MAX_TOKENS="${MAX_TOKENS:-512}"
WARMUP_TOKENS="${WARMUP_TOKENS:-128}"
RUNS_PER_PROMPT="${RUNS_PER_PROMPT:-1}"
MATRIX_ENABLE_PREFIX_CACHING="${MATRIX_ENABLE_PREFIX_CACHING:-1}"
RESULT_DIR="${RESULT_DIR:-/tmp/glm52-mtp-dcp-matrix-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${RESULT_DIR}/env" "${RESULT_DIR}/logs"

if [[ ! -f "${BASE_ENV}" ]]; then
  echo "Missing BASE_ENV=${BASE_ENV}" >&2
  exit 2
fi

make_env() {
  local dcp="$1"
  local mtp="$2"
  local name="glm52-matrix-dcp${dcp}-mtp${mtp}-16k"
  local env_file="${RESULT_DIR}/env/${name}.env"
  cp "${BASE_ENV}" "${env_file}"
  perl -0pi -e "s/^SERVED_MODEL_NAME=.*/SERVED_MODEL_NAME=${name}/m; s/^DCP_SIZE=.*/DCP_SIZE=${dcp}/m; s/^MAX_MODEL_LEN=.*/MAX_MODEL_LEN=${MAX_MODEL_LEN}/m; s/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}/m; s/^KV_CACHE_MEMORY_BYTES=.*/KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES}/m; s/^KZ_KV_DIAG=.*/KZ_KV_DIAG=1/m" "${env_file}"
  if grep -q '^LOG_FILE=' "${env_file}"; then
    perl -0pi -e "s#^LOG_FILE=.*#LOG_FILE=/tmp/${name}.log#m" "${env_file}"
  else
    printf '\nLOG_FILE=/tmp/%s.log\n' "${name}" >>"${env_file}"
  fi
  if grep -q '^ENABLE_PREFIX_CACHING=' "${env_file}"; then
    perl -0pi -e "s/^ENABLE_PREFIX_CACHING=.*/ENABLE_PREFIX_CACHING=${MATRIX_ENABLE_PREFIX_CACHING}/m" "${env_file}"
  else
    printf 'ENABLE_PREFIX_CACHING=%s\n' "${MATRIX_ENABLE_PREFIX_CACHING}" >>"${env_file}"
  fi
  if [[ "${mtp}" == "0" ]]; then
    perl -0pi -e "s/^ENABLE_MTP=.*/ENABLE_MTP=0/m; s/^NUM_SPECULATIVE_TOKENS=.*/NUM_SPECULATIVE_TOKENS=0/m" "${env_file}"
  else
    perl -0pi -e "s/^ENABLE_MTP=.*/ENABLE_MTP=1/m; s/^NUM_SPECULATIVE_TOKENS=.*/NUM_SPECULATIVE_TOKENS=${mtp}/m" "${env_file}"
  fi
  echo "${env_file}"
}

wait_ready() {
  local model="$1"
  local url="http://${HEAD_IP}:${PORT}/v1/models"
  python3 - "$url" "$model" <<'PY'
import json, sys, time, urllib.request
url, expected = sys.argv[1], sys.argv[2]
deadline = time.time() + 3600
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = json.loads(resp.read())
        ids = [item.get("id") for item in body.get("data", [])]
        last = ids
        if expected in ids:
            print(json.dumps(body))
            sys.exit(0)
    except Exception as exc:
        last = repr(exc)
    time.sleep(10)
print(f"Timed out waiting for {expected}; last={last}", file=sys.stderr)
sys.exit(124)
PY
}

capture_worker_env() {
  local model="$1"
  local out="${RESULT_DIR}/logs/${model}.worker-env.txt"
  {
    echo "== head =="
    docker exec "${HEAD_NAME}" bash -lc 'for p in $(pgrep -f "ray::RayWorkerProc" | sort -n); do echo "-- pid=$p --"; tr "\0" "\n" </proc/$p/environ | grep -E "^(VLLM_DCP|VLLM_MTP|NCCL_|GLOO_|B12X_|CUDA_DEVICE_MAX_CONNECTIONS|VLLM_USE_B12X|VLLM_USE_FLASHINFER|KZ_KV_DIAG|VLLM_KZ_TRIM)" | sort; done' || true
    for ip in "${WORKER_IPS[@]}"; do
      echo "== ${ip} =="
      ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5 "${ip}" 'docker exec glm-dark-worker bash -lc '\''for p in $(pgrep -f "ray::RayWorkerProc" | sort -n); do echo "-- pid=$p --"; tr "\0" "\n" </proc/$p/environ | grep -E "^(VLLM_DCP|VLLM_MTP|NCCL_|GLOO_|B12X_|CUDA_DEVICE_MAX_CONNECTIONS|VLLM_USE_B12X|VLLM_USE_FLASHINFER|KZ_KV_DIAG|VLLM_KZ_TRIM)" | sort; done'\''' || true
    done
  } >"${out}" 2>&1
}

run_benchmarks() {
  local model="$1"
  local out_jsonl="${RESULT_DIR}/${model}.jsonl"
  python3 - "$model" "http://${HEAD_IP}:${PORT}/v1/completions" "${out_jsonl}" "${MAX_TOKENS}" "${WARMUP_TOKENS}" "${RUNS_PER_PROMPT}" <<'PY'
import json, sys, time, urllib.request
model, url, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
max_tokens, warmup_tokens, runs_per_prompt = map(int, sys.argv[4:7])

seed = (
    "I am by birth a Genevese, and my family is one of the most distinguished of that republic. "
    "My ancestors had been for many years counsellors and syndics, and my father had filled several public situations with honour and reputation. "
    "He was respected by all who knew him for his integrity and indefatigable attention to public business. "
    "As the minuteness of circumstances is often useful in preserving the chain of causes and effects, I shall be particular in this account. "
    "The winter wind moved over the mountains and the lake, and every household tale became a meditation on ambition, responsibility, and the cost of knowledge. "
)
body = []
while len("".join(body)) < 48000:
    body.append(seed)
long_context = "".join(body)[:48000]

prompts = [
    ("warmup", "Write Python code to efficiently print prime numbers from 0 to 1,000,000,000. Focus on algorithmic efficiency and practical constraints.", warmup_tokens),
    ("zero_ctx_kanban", "Write me a dead simple FastAPI + Pydantic + SQLite + SQLAlchemy + React kanban implementation. Include the backend models, API routes, frontend components, and minimal run instructions.", max_tokens),
    ("resident_12k_summary", "Below is a long Frankenstein-style public-domain context. Summarize the major themes, character motivations, and causal chain in a concise technical brief.\n\n" + long_context + "\n\nSummary:", max_tokens),
]

def request(label, prompt, tokens, run):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": tokens,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()
    elapsed = time.perf_counter() - t0
    body = json.loads(raw)
    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    prompt_tokens = usage.get("prompt_tokens") or 0
    rec = {
        "model": model,
        "label": label,
        "run": run,
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "completion_tok_s": completion_tokens / elapsed if elapsed else None,
        "created": body.get("created"),
        "finish_reason": ((body.get("choices") or [{}])[0]).get("finish_reason"),
    }
    print(json.dumps(rec, sort_keys=True), flush=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")

for label, prompt, tokens in prompts:
    count = 1 if label == "warmup" else runs_per_prompt
    for run in range(1, count + 1):
        request(label, prompt, tokens, run)
PY
}

capture_server_metrics() {
  local model="$1"
  docker exec "${HEAD_NAME}" bash -lc "grep -E 'GPU KV cache size|Maximum concurrency|Available KV cache memory|SpecDecoding metrics|Avg prompt throughput|Avg generation throughput|ERROR|Traceback|RuntimeError|ValueError' /tmp/${model}.log | tail -n 240" >"${RESULT_DIR}/logs/${model}.server-metrics.log" 2>&1 || true
  docker exec "${HEAD_NAME}" bash -lc "tail -n 240 /tmp/${model}.log" >"${RESULT_DIR}/logs/${model}.tail.log" 2>&1 || true
}

summary_file="${RESULT_DIR}/summary.tsv"
if [[ ! -f "${summary_file}" ]]; then
  printf 'model\tdcp\tmtp\tlabel\trun\tprompt_tokens\tcompletion_tokens\telapsed_s\tcompletion_tok_s\n' >"${summary_file}"
fi

for dcp in "${DCP_LIST[@]}"; do
  for mtp in "${MTP_LIST[@]}"; do
    model="glm52-matrix-dcp${dcp}-mtp${mtp}-16k"
    env_file="$(make_env "${dcp}" "${mtp}")"
    echo "==== ${model} ====" | tee -a "${RESULT_DIR}/matrix.log"
    ENV_FILE="${env_file}" "${LAUNCHER}" 2>&1 | tee "${RESULT_DIR}/logs/${model}.launch.log"
    wait_ready "${model}" >"${RESULT_DIR}/logs/${model}.models.json"
    capture_worker_env "${model}"
    run_benchmarks "${model}" 2>&1 | tee "${RESULT_DIR}/logs/${model}.bench.log"
    capture_server_metrics "${model}"
    python3 - "${RESULT_DIR}/${model}.jsonl" "${summary_file}" "${dcp}" "${mtp}" <<'PY'
import json, sys
src, dst, dcp, mtp = sys.argv[1:]
with open(src, encoding="utf-8") as f, open(dst, "a", encoding="utf-8") as out:
    for line in f:
        r = json.loads(line)
        out.write("\t".join(str(x) for x in [r["model"], dcp, mtp, r["label"], r["run"], r["prompt_tokens"], r["completion_tokens"], f'{r["elapsed_s"]:.3f}', f'{r["completion_tok_s"]:.3f}']) + "\n")
PY
  done
done

echo "RESULT_DIR=${RESULT_DIR}"
echo "SUMMARY=${summary_file}"
