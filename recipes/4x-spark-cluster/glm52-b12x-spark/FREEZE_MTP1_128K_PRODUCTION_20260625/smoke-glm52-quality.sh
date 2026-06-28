#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

# shellcheck source=glm52-mtp1-dcp4-128k.env
source ./glm52-mtp1-dcp4-128k.env

BASE_URL=${BASE_URL:-http://192.168.100.1:18089/v1}
MODEL=${MODEL:-${SERVED_MODEL_NAME}}
TIMEOUT=${TIMEOUT:-240}
ENABLE_THINKING=${ENABLE_THINKING:-0}

BASE_URL="${BASE_URL}" MODEL="${MODEL}" TIMEOUT="${TIMEOUT}" ENABLE_THINKING="${ENABLE_THINKING}" python3 - <<'PY'
import json
import os
import re
import sys
import time
import urllib.request

base_url = os.environ["BASE_URL"].rstrip("/")
model = os.environ["MODEL"]
timeout = int(os.environ["TIMEOUT"])
enable_thinking = os.environ.get("ENABLE_THINKING", "0") not in {"0", "false", "False", "no", "NO"}

tests = [
    {
        "name": "exact-token",
        "messages": [
            {"role": "system", "content": "Follow the user's instruction exactly."},
            {"role": "user", "content": "Reply with exactly this token and nothing else: SPARK-GLM52-OK"},
        ],
        "max_tokens": 24,
        "checks": [lambda text: "SPARK-GLM52-OK" in text],
    },
    {
        "name": "arithmetic",
        "messages": [
            {"role": "system", "content": "Answer with only the requested final value."},
            {"role": "user", "content": "What is 17 * 23? Answer with only the integer."},
        ],
        "max_tokens": 24,
        "checks": [lambda text: re.search(r"\b391\b", text) is not None],
    },
    {
        "name": "code-shape",
        "messages": [
            {"role": "system", "content": "Output only valid Python code."},
            {"role": "user", "content": "Write a Python function named add that returns a + b."},
        ],
        "max_tokens": 96,
        "checks": [
            lambda text: "def add" in text,
            lambda text: "return" in text,
            lambda text: ("a + b" in text) or ("a+b" in text),
        ],
    },
]

all_completion_tokens = 0
all_elapsed = 0.0
for test in tests:
    payload = {
        "model": model,
        "messages": test["messages"],
        "max_tokens": test["max_tokens"],
        "temperature": 0.0,
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
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or max(1, len(text.split())))
    all_completion_tokens += completion_tokens
    all_elapsed += elapsed
    preview = text.replace("\n", "\\n")[:240]
    if not text.strip():
        print(f"FAIL {test['name']}: empty response", file=sys.stderr)
        raise SystemExit(1)
    if not all(check(text) for check in test["checks"]):
        print(f"FAIL {test['name']}: response did not pass checks", file=sys.stderr)
        print(f"PREVIEW {preview}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"PASS {test['name']}: {completion_tokens} tokens in {elapsed:.3f}s "
        f"({completion_tokens / elapsed:.3f} tok/s) :: {preview}"
    )

if all_elapsed > 0:
    print(
        f"PASS aggregate-quality-smoke: {all_completion_tokens} completion tokens "
        f"in {all_elapsed:.3f}s ({all_completion_tokens / all_elapsed:.3f} tok/s)"
    )
PY
