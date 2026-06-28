#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="$PWD/logs/shape-matrix-${STAMP}"
mkdir -p "$OUTDIR"

ENV_FILES=(
  "glm52-dcp4-mtp1-128k.env"
  "glm52-dcp4-mtp2-120k-loose.env"
)

SSH=(ssh -i /etc/kamiwaza/ssl/cluster.key -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5)
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

fetch_corpus() {
  mkdir -p "$PWD/corpus"
  if [ ! -s "$PWD/corpus/frankenstein.txt" ]; then
    curl -fL --retry 3 https://www.gutenberg.org/files/84/84-0.txt -o "$PWD/corpus/frankenstein.txt"
  fi
  if [ ! -s "$PWD/corpus/war_and_peace.txt" ]; then
    curl -fL --retry 3 https://www.gutenberg.org/files/2600/2600-0.txt -o "$PWD/corpus/war_and_peace.txt"
  fi
  if [ ! -s "$PWD/corpus/bible.txt" ]; then
    curl -fL --retry 3 https://www.gutenberg.org/files/10/10-0.txt -o "$PWD/corpus/bible.txt"
  fi
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

write_client() {
  cat > "$OUTDIR/shape_matrix_client.py" <<'PY'
import csv
import json
import os
import re
import sys
import time
import urllib.error
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
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


def load_corpus(root):
    corpus_dir = Path(root) / "corpus"
    return {
        "frankenstein": strip_gutenberg((corpus_dir / "frankenstein.txt").read_text(errors="replace")),
        "warpeace": strip_gutenberg((corpus_dir / "war_and_peace.txt").read_text(errors="replace")),
        "bible": strip_gutenberg((corpus_dir / "bible.txt").read_text(errors="replace")),
    }


def source_slice(text, approx_chars, offset_frac):
    text = re.sub(r"\s+", " ", text).strip()
    if approx_chars >= len(text):
        return text
    start_max = max(0, len(text) - approx_chars - 1)
    start = int(start_max * offset_frac)
    return text[start:start + approx_chars]


class Client:
    def __init__(self, base_url, model, result_dir):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.result_dir = Path(result_dir)
        self.tokenize_available = None

    def post_json(self, path, payload, timeout=7200):
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def token_count(self, prompt):
        if self.tokenize_available is False:
            return max(1, int(len(prompt) / 4.0))
        try:
            data = self.post_json("/tokenize", {"model": self.model, "prompt": prompt}, timeout=300)
            self.tokenize_available = True
            if isinstance(data.get("count"), int):
                return data["count"]
            for key in ("tokens", "token_ids", "input_ids"):
                if isinstance(data.get(key), list):
                    return len(data[key])
        except Exception:
            self.tokenize_available = False
        return max(1, int(len(prompt) / 4.0))

    def fit_prompt(self, source_text, target_tokens, builder, offset_frac):
        lo = 1_000
        hi = min(len(re.sub(r"\s+", " ", source_text)), max(8_000, int(target_tokens * 5.4)))
        best_prompt = builder(source_slice(source_text, min(hi, int(target_tokens * 4.0)), offset_frac))
        best_count = self.token_count(best_prompt)
        for _ in range(16):
            mid = (lo + hi) // 2
            prompt = builder(source_slice(source_text, mid, offset_frac))
            count = self.token_count(prompt)
            if count <= target_tokens:
                best_prompt, best_count = prompt, count
                lo = mid + 1
            else:
                hi = mid - 1
        return best_prompt, best_count

    def stream_completion(self, case, prompt, max_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
        }
        req = urllib.request.Request(
            self.base_url + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        out_parts = []
        usage = None
        t0 = time.perf_counter()
        last_progress = t0
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw in resp:
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
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices:
                    text = choices[0].get("text") or ""
                    if text:
                        out_parts.append(text)
                now = time.perf_counter()
                if now - last_progress >= 60:
                    chars = sum(len(p) for p in out_parts)
                    print(f"[progress] {case} seconds={now - t0:.1f} output_chars={chars}", flush=True)
                    last_progress = now
        seconds = time.perf_counter() - t0
        text = "".join(out_parts)
        return text, usage, seconds


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


def translate_builder(label):
    def build(excerpt):
        return (
            f"Translate the following unique public-domain English excerpt labeled {label} into German. "
            "Preserve paragraphs where practical. Output only the German translation.\n\n"
            "<english>\n"
            f"{excerpt}\n"
            "</english>\n\nGerman translation:\n"
        )
    return build


def build_cases(client, corpus):
    cases = []
    codegen = [
        ("codegen_0k_fastapi_jobs", "Write a compact but complete FastAPI + Pydantic + SQLite job queue service with a React status page. Include schema, API routes, and the main React component.", 1024),
        ("codegen_0k_rust_logmerge", "Write a Rust CLI that merges timestamped JSONL logs from many files, keeps stable ordering, and exposes useful tests. Keep it practical.", 1024),
        ("codegen_0k_sqlalchemy_audit", "Write a Python SQLAlchemy 2.0 audit-log mixin and migration sketch for a multi-tenant SaaS app. Include concise example usage.", 1024),
    ]
    for name, prompt, max_tokens in codegen:
        cases.append({
            "name": name,
            "kind": "codegen",
            "target_prompt_tokens": 0,
            "prompt": prompt,
            "prompt_tokens_est": client.token_count(prompt),
            "max_tokens": max_tokens,
        })

    summarize_specs = [
        ("summarize_16k_frankenstein", "frankenstein", 16_000, 0.18),
        ("summarize_32k_warpeace", "warpeace", 32_000, 0.27),
        ("summarize_64k_bible", "bible", 64_000, 0.41),
        ("summarize_112k_warpeace", "warpeace", 112_000, 0.62),
    ]
    for name, source, target, offset in summarize_specs:
        prompt, count = client.fit_prompt(corpus[source], target, summarize_builder(name), offset)
        cases.append({
            "name": name,
            "kind": "summarize",
            "source": source,
            "target_prompt_tokens": target,
            "prompt": prompt,
            "prompt_tokens_est": count,
            "max_tokens": 256,
        })

    translate_specs = [
        ("translate_12k5_frankenstein", "frankenstein", 12_500, 0.56),
        ("translate_25k_warpeace", "warpeace", 25_000, 0.72),
        ("translate_50k_bible", "bible", 50_000, 0.32),
    ]
    for name, source, target, offset in translate_specs:
        prompt, count = client.fit_prompt(corpus[source], target, translate_builder(name), offset)
        cases.append({
            "name": name,
            "kind": "translate",
            "source": source,
            "target_prompt_tokens": target,
            "prompt": prompt,
            "prompt_tokens_est": count,
            "max_tokens": target,
        })
    return cases


def write_csv(path, records):
    fields = [
        "setup", "case", "kind", "target_prompt_tokens", "prompt_tokens",
        "completion_tokens", "total_tokens", "max_tokens", "seconds",
        "completion_tps", "total_tps", "finish_reason", "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})


def main():
    if len(sys.argv) != 5:
        print("usage: shape_matrix_client.py ROOT OUTDIR SETUP MODEL", file=sys.stderr)
        return 2
    root, outdir, setup, model = sys.argv[1:]
    result_dir = Path(outdir)
    result_dir.mkdir(parents=True, exist_ok=True)
    client = Client("http://192.168.100.1:18089", model, result_dir)
    corpus = load_corpus(root)
    cases = build_cases(client, corpus)
    (result_dir / "cases_manifest.json").write_text(json.dumps([
        {k: v for k, v in c.items() if k != "prompt"} for c in cases
    ], indent=2))

    records = []
    jsonl_path = result_dir / "results.jsonl"
    with open(jsonl_path, "a") as jf:
        for idx, case in enumerate(cases, 1):
            print(
                f"[case {idx}/{len(cases)}] {setup} {case['name']} "
                f"prompt_est={case['prompt_tokens_est']} max_tokens={case['max_tokens']}",
                flush=True,
            )
            rec = {
                "setup": setup,
                "case": case["name"],
                "kind": case["kind"],
                "source": case.get("source", ""),
                "target_prompt_tokens": case["target_prompt_tokens"],
                "prompt_tokens_est": case["prompt_tokens_est"],
                "max_tokens": case["max_tokens"],
                "error": "",
            }
            try:
                text, usage, seconds = client.stream_completion(case["name"], case["prompt"], case["max_tokens"])
                output_path = result_dir / f"{case['name']}.output.txt"
                output_path.write_text(text, errors="replace")
                completion_tokens = None
                prompt_tokens = case["prompt_tokens_est"]
                total_tokens = None
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")
                if completion_tokens is None:
                    completion_tokens = client.token_count(text)
                    total_tokens = prompt_tokens + completion_tokens
                rec.update({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "seconds": seconds,
                    "completion_tps": completion_tokens / seconds if seconds else 0.0,
                    "total_tps": total_tokens / seconds if seconds else 0.0,
                    "finish_reason": "unknown_stream",
                    "output_path": str(output_path),
                    "output_prefix": text[:240].replace("\n", " "),
                })
                print(
                    f"[result] {setup} {case['name']} prompt={prompt_tokens} "
                    f"completion={completion_tokens} seconds={seconds:.3f} "
                    f"completion_tps={rec['completion_tps']:.3f}",
                    flush=True,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                rec.update({"error": f"HTTP {exc.code}: {body[:1000]}"})
                print(f"[error] {setup} {case['name']} HTTP {exc.code}: {body[:300]}", flush=True)
            except Exception as exc:
                rec.update({"error": repr(exc)})
                print(f"[error] {setup} {case['name']} {exc!r}", flush=True)
            records.append(rec)
            jf.write(json.dumps(rec) + "\n")
            jf.flush()
            write_csv(result_dir / "results.csv", records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
}

trap cleanup_deployment EXIT

fetch_corpus
write_client
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
    python3 "$OUTDIR/shape_matrix_client.py" "$PWD" "$setup_out" "$setup_name" "$setup_name" | tee "$setup_out/client.log"
    docker exec glm-dark-head bash -lc "cat /tmp/${setup_name}.log 2>/dev/null || true" > "$setup_out/vllm.log" || true
  else
    echo "[setup failed] $setup_name"
  fi
  cleanup_deployment
  drain_swap_cache_all
done

python3 - "$OUTDIR" <<'PY' | tee "$OUTDIR/summary.csv"
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob("*/results.csv")):
    with open(path, newline="") as f:
        rows.extend(csv.DictReader(f))
if not rows:
    print("setup,case,kind,target_prompt_tokens,prompt_tokens,completion_tokens,seconds,completion_tps,error")
    raise SystemExit(0)
fields = ["setup", "case", "kind", "target_prompt_tokens", "prompt_tokens", "completion_tokens", "seconds", "completion_tps", "error"]
writer = csv.DictWriter(sys.stdout, fieldnames=fields)
writer.writeheader()
for row in rows:
    writer.writerow({k: row.get(k, "") for k in fields})
PY

echo "OUTDIR=$OUTDIR"
