#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

# shellcheck source=glm52-mtp1-dcp4-128k.env
source ./glm52-mtp1-dcp4-128k.env

BASE_URL=${BASE_URL:-http://192.168.100.1:18089/v1}
MODEL=${MODEL:-${SERVED_MODEL_NAME}}
MAX_TOKENS=${MAX_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0.0}
TIMEOUT=${TIMEOUT:-600}
ENABLE_THINKING=${ENABLE_THINKING:-0}
PROMPT=${PROMPT:-Write a complete Python module implementing an LRU cache with get, put, delete, clear, iteration over keys, and a small self-test section. Include comments but no markdown.}

BASE_URL="${BASE_URL}" \
MODEL="${MODEL}" \
MAX_TOKENS="${MAX_TOKENS}" \
TEMPERATURE="${TEMPERATURE}" \
TIMEOUT="${TIMEOUT}" \
ENABLE_THINKING="${ENABLE_THINKING}" \
PROMPT="${PROMPT}" \
python3 - <<'PY'
import json
import os
import re
import time
import urllib.request

base_url = os.environ["BASE_URL"].rstrip("/")
model = os.environ["MODEL"]
max_tokens = int(os.environ["MAX_TOKENS"])
temperature = float(os.environ["TEMPERATURE"])
timeout = int(os.environ["TIMEOUT"])
enable_thinking = os.environ.get("ENABLE_THINKING", "0") not in {"0", "false", "False", "no", "NO"}
prompt = os.environ["PROMPT"]

metric_names = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "gen_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
}

metric_re = re.compile(r'^(vllm:[^{\s]+)(?:\{[^}]*model_name="' + re.escape(model) + r'"[^}]*\})?\s+([0-9.eE+-]+)$')

def fetch_metrics():
    with urllib.request.urlopen(base_url.rsplit("/", 1)[0] + "/metrics", timeout=30) as response:
        text = response.read().decode()
    values = {key: 0.0 for key in metric_names}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = metric_re.match(line)
        if not m:
            continue
        metric_name, value = m.groups()
        for key, wanted in metric_names.items():
            if metric_name == wanted:
                values[key] += float(value)
    return values

before = fetch_metrics()
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a concise senior Python engineer. Output only the requested code."},
        {"role": "user", "content": prompt},
    ],
    "max_tokens": max_tokens,
    "temperature": temperature,
    "chat_template_kwargs": {"enable_thinking": enable_thinking, "thinking": enable_thinking},
}
req = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
start = time.perf_counter()
with urllib.request.urlopen(req, timeout=timeout) as response:
    data = json.loads(response.read().decode())
elapsed = time.perf_counter() - start
after = fetch_metrics()
choice = (data.get("choices") or [{}])[0]
message = choice.get("message") or {}
text = message.get("content") or ""
usage = data.get("usage") or {}
completion_tokens = int(usage.get("completion_tokens") or max(1, len(text.split())))
prompt_tokens = int(usage.get("prompt_tokens") or 0)
finish_reason = choice.get("finish_reason")

delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in metric_names}
accepted = delta["accepted_tokens"]
draft_tokens = delta["draft_tokens"]
acceptance = accepted / draft_tokens if draft_tokens > 0 else 0.0
client_tps = completion_tokens / elapsed if elapsed > 0 else 0.0
metric_gen_tps = delta["gen_tokens"] / elapsed if elapsed > 0 else 0.0

preview = text.replace("\n", "\\n")[:260]
print(f"model={model}")
print(f"finish_reason={finish_reason}")
print(f"prompt_tokens={prompt_tokens}")
print(f"completion_tokens={completion_tokens}")
print(f"elapsed_s={elapsed:.3f}")
print(f"client_completion_tps={client_tps:.3f}")
print(f"metric_generation_delta={delta['gen_tokens']:.0f}")
print(f"metric_generation_tps={metric_gen_tps:.3f}")
print(f"spec_drafts_delta={delta['drafts']:.0f}")
print(f"spec_draft_tokens_delta={draft_tokens:.0f}")
print(f"spec_accepted_tokens_delta={accepted:.0f}")
print(f"spec_acceptance_ratio={acceptance:.3f}")
print(f"preview={preview}")
PY
