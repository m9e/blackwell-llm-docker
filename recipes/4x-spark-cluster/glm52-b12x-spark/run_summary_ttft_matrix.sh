#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="$PWD/logs/summary-ttft-${STAMP}"
mkdir -p "$OUTDIR"

ENV_FILES=(
  "glm52-dcp4-mtp1-128k.env"
  "glm52-dcp4-mtp2-120k-loose.env"
)

SSH=(ssh -i /etc/kamiwaza/ssl/cluster.key -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=no -o ConnectTimeout=5)
HOST_IPS=(192.168.100.2 192.168.100.3 192.168.100.4)

echo "OUTDIR=$OUTDIR"

cleanup_deployment() {
  docker rm -f glm-dark-head >/dev/null 2>&1 || true
  for ip in "${HOST_IPS[@]}"; do
    "${SSH[@]}" "$ip" 'docker rm -f glm-dark-worker >/dev/null 2>&1 || true' >/dev/null 2>&1 || true
  done
}

drain_swap_cache_all() {
  echo "[drain] local"
  sudo -n swapoff -a && sudo -n swapon -a && sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
  for ip in "${HOST_IPS[@]}"; do
    echo "[drain] $ip"
    "${SSH[@]}" "$ip" "sudo -n swapoff -a && sudo -n swapon -a && sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" >/dev/null 2>&1 || true
  done
}

wait_ready() {
  local served=$1
  local outdir=$2
  local log="/tmp/${served}.log"
  for _ in $(seq 1 160); do
    if docker exec glm-dark-head test -f "$log" >/dev/null 2>&1; then
      if docker exec glm-dark-head grep -q 'Application startup complete' "$log"; then
        echo "[ready] $served"
        docker exec glm-dark-head grep -E 'GPU KV cache size|Maximum concurrency|Available KV cache memory|Application startup complete' "$log" \
          | tail -40 | tee "$outdir/readiness.log"
        return 0
      fi
      if docker exec glm-dark-head grep -qE 'EngineCore failed|Traceback|ValueError|CUDA out of memory' "$log"; then
        echo "[failed] $served"
        docker exec glm-dark-head grep -E 'EngineCore failed|Traceback|ValueError|CUDA out of memory|Free memory|Available KV cache memory|GPU KV cache size|Maximum concurrency' "$log" \
          | tail -160 | tee "$outdir/readiness.log"
        return 1
      fi
    fi
    docker exec glm-dark-head bash -lc 'tail -n 120 /tmp/ray-vllm-head/session_latest/logs/worker-*.err 2>/dev/null | grep -E "Loading safetensors|Loaded weights|Available KV cache|GPU KV cache|Maximum concurrency" | tail -6 || true' \
      | tee -a "$outdir/load_progress.log"
    sleep 30
  done
  echo "[timeout] $served"
  return 2
}

cat > "$OUTDIR/summary_ttft_client.py" <<'PY'
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path


def strip_gutenberg(text):
    text = text.replace("\r\n", "\n")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    return re.sub(r"\n{4,}", "\n\n", text).strip()


def source_slice(text, approx_chars, offset_frac):
    text = re.sub(r"\s+", " ", text).strip()
    if approx_chars >= len(text):
        return text
    start = int(max(0, len(text) - approx_chars - 1) * offset_frac)
    return text[start:start + approx_chars]


class Client:
    def __init__(self, model):
        self.model = model
        self.base = "http://192.168.100.1:18089"
        self.tokenize_available = None

    def post_json(self, path, payload, timeout=3600):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def token_count(self, text):
        if self.tokenize_available is False:
            return max(1, int(len(text) / 4.0))
        try:
            data = self.post_json("/tokenize", {"model": self.model, "prompt": text}, timeout=300)
            self.tokenize_available = True
            if isinstance(data.get("count"), int):
                return data["count"]
            for key in ("tokens", "token_ids", "input_ids"):
                if isinstance(data.get(key), list):
                    return len(data[key])
        except Exception:
            self.tokenize_available = False
        return max(1, int(len(text) / 4.0))

    def fit_prompt(self, source_text, target_tokens, builder, offset_frac):
        clean = re.sub(r"\s+", " ", source_text).strip()
        lo = 1000
        hi = min(len(clean), max(8000, int(target_tokens * 5.4)))
        best_prompt = builder(source_slice(clean, min(hi, int(target_tokens * 4.0)), offset_frac))
        best_count = self.token_count(best_prompt)
        for _ in range(16):
            mid = (lo + hi) // 2
            prompt = builder(source_slice(clean, mid, offset_frac))
            count = self.token_count(prompt)
            if count <= target_tokens:
                best_prompt, best_count = prompt, count
                lo = mid + 1
            else:
                hi = mid - 1
        return best_prompt, best_count

    def stream_completion_with_ttft(self, case_name, prompt, max_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
        }
        req = urllib.request.Request(
            self.base + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        parts = []
        t0 = time.perf_counter()
        first_token_time = None
        first_byte_time = None
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                now = time.perf_counter()
                if first_byte_time is None:
                    first_byte_time = now
                line = raw.decode(errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if text and first_token_time is None:
                    first_token_time = now
                if text:
                    parts.append(text)
        t1 = time.perf_counter()
        output = "".join(parts)
        completion_tokens = self.token_count(output)
        ttft = None if first_token_time is None else first_token_time - t0
        first_byte = None if first_byte_time is None else first_byte_time - t0
        decode_seconds = None if ttft is None else max(0.000001, t1 - first_token_time)
        return {
            "seconds": t1 - t0,
            "ttft_seconds": ttft,
            "first_byte_seconds": first_byte,
            "decode_seconds_after_ttft": decode_seconds,
            "completion_tokens": completion_tokens,
            "completion_tps_e2e": completion_tokens / (t1 - t0),
            "completion_tps_after_ttft": None if decode_seconds is None else completion_tokens / decode_seconds,
            "output_prefix": output[:240].replace("\n", " "),
        }


def summarize_builder(label):
    def build(excerpt):
        return (
            f"You are reading a unique public-domain excerpt labeled {label}.\n"
            "Summarize the excerpt briefly in 8-12 bullets. Be precise and do not continue beyond the summary.\n\n"
            "<excerpt>\n"
            f"{excerpt}\n"
            "</excerpt>\n\nBrief summary:\n"
        )
    return build


def main():
    root, outdir, setup, model = sys.argv[1:]
    out = Path(outdir)
    corpus_dir = Path(root) / "corpus"
    corpus = {
        "frankenstein": strip_gutenberg((corpus_dir / "frankenstein.txt").read_text(errors="replace")),
        "warpeace": strip_gutenberg((corpus_dir / "war_and_peace.txt").read_text(errors="replace")),
        "bible": strip_gutenberg((corpus_dir / "bible.txt").read_text(errors="replace")),
    }
    specs = [
        ("summarize_16k_frankenstein", "frankenstein", 16_000, 0.18),
        ("summarize_32k_warpeace", "warpeace", 32_000, 0.27),
        ("summarize_64k_bible", "bible", 64_000, 0.41),
        ("summarize_112k_warpeace", "warpeace", 112_000, 0.62),
    ]
    client = Client(model)
    rows = []
    with open(out / "summary_ttft.jsonl", "a") as jf:
        for name, source, target, offset in specs:
            prompt, prompt_tokens = client.fit_prompt(corpus[source], target, summarize_builder(name), offset)
            print(f"[case] {setup} {name} prompt={prompt_tokens}", flush=True)
            rec = {
                "setup": setup,
                "case": name,
                "source": source,
                "target_prompt_tokens": target,
                "prompt_tokens": prompt_tokens,
                "max_tokens": 256,
                "error": "",
            }
            try:
                rec.update(client.stream_completion_with_ttft(name, prompt, 256))
                print(
                    "[result] {setup} {case} prompt={prompt_tokens} completion={completion_tokens} "
                    "seconds={seconds:.3f} ttft={ttft_seconds:.3f} decode_after_ttft={decode_seconds_after_ttft:.3f} "
                    "e2e_tps={completion_tps_e2e:.3f} decode_tps={completion_tps_after_ttft:.3f}".format(**rec),
                    flush=True,
                )
            except Exception as exc:
                rec["error"] = repr(exc)
                print(f"[error] {setup} {name} {exc!r}", flush=True)
            rows.append(rec)
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            with open(out / "summary_ttft.csv", "w", newline="") as f:
                fields = [
                    "setup", "case", "source", "target_prompt_tokens", "prompt_tokens",
                    "completion_tokens", "seconds", "ttft_seconds", "decode_seconds_after_ttft",
                    "completion_tps_e2e", "completion_tps_after_ttft", "error",
                ]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in fields})


if __name__ == "__main__":
    main()
PY

trap cleanup_deployment EXIT
cleanup_deployment

for env_file in "${ENV_FILES[@]}"; do
  if [ ! -f "$env_file" ]; then
    echo "missing env file: $env_file" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$env_file"
  setup_name="${SERVED_MODEL_NAME:?SERVED_MODEL_NAME missing}"
  setup_out="$OUTDIR/$setup_name"
  mkdir -p "$setup_out"
  cp "$env_file" "$setup_out/"
  echo "[setup] $setup_name env=$env_file"
  cleanup_deployment
  drain_swap_cache_all
  ENV_FILE="$PWD/$env_file" PATCH_DIAGNOSTICS=1 ./launch-glm52-mtp3-dcp4-128k.sh | tee "$setup_out/launch.log"
  if wait_ready "$setup_name" "$setup_out"; then
    python3 "$OUTDIR/summary_ttft_client.py" "$PWD" "$setup_out" "$setup_name" "$setup_name" | tee "$setup_out/client.log"
    docker exec glm-dark-head bash -lc "cat /tmp/${setup_name}.log 2>/dev/null || true" > "$setup_out/vllm.log" || true
  fi
  cleanup_deployment
  drain_swap_cache_all
done

python3 - "$OUTDIR" <<'PY' | tee "$OUTDIR/summary_ttft_all.csv"
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob("*/summary_ttft.csv")):
    with open(path, newline="") as f:
        rows.extend(csv.DictReader(f))
fields = [
    "setup", "case", "target_prompt_tokens", "prompt_tokens", "completion_tokens",
    "seconds", "ttft_seconds", "decode_seconds_after_ttft",
    "completion_tps_e2e", "completion_tps_after_ttft", "error",
]
writer = csv.DictWriter(sys.stdout, fieldnames=fields)
writer.writeheader()
for row in rows:
    writer.writerow({k: row.get(k, "") for k in fields})
PY

echo "OUTDIR=$OUTDIR"
