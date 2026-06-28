#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 ENV_FILE TAG" >&2
  exit 2
fi

ENV_FILE=$1
TAG=$2
cd "$(dirname "$0")"
if [ ! -f "$ENV_FILE" ]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
SERVED=${SERVED_MODEL_NAME:?SERVED_MODEL_NAME missing from env}
STAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="$PWD/logs/live-swap-${STAMP}-${TAG}"
mkdir -p "$OUTDIR"

SSH=(ssh -i /etc/kamiwaza/ssl/cluster.key -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5)
SCP=(scp -i /etc/kamiwaza/ssl/cluster.key -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes -o ConnectTimeout=5)
HOSTS=("relic:127.0.0.1" "soulkiller:192.168.100.2" "cynosure:192.168.100.3" "blackwall:192.168.100.4")

echo "OUTDIR=$OUTDIR"
echo "ENV_FILE=$ENV_FILE"
echo "SERVED=$SERVED"

# Ensure helpers are current.
cp kw_swap_diag.sh /tmp/kw_swap_diag.sh
chmod +x /tmp/kw_swap_diag.sh
cat > /tmp/kw_gpu_pid_swap.py <<'PY'
import subprocess
try:
    out = subprocess.check_output([
        'nvidia-smi',
        '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits',
    ], text=True, stderr=subprocess.DEVNULL)
except Exception:
    out = ''
print(out.strip() or 'no nvidia compute pids')
for line in out.splitlines():
    parts = [x.strip() for x in line.split(',')]
    if not parts or not parts[0].isdigit():
        continue
    pid = parts[0]
    status = {}
    try:
        with open(f'/proc/{pid}/status') as f:
            for l in f:
                if l.startswith(('Name:', 'VmRSS:', 'VmSwap:')):
                    k, v = l.split(':', 1)
                    status[k] = v.strip()
    except FileNotFoundError:
        continue
    majflt = '?'
    try:
        s = open(f'/proc/{pid}/stat').read()
        rest = s.rsplit(') ', 1)[1].split()
        majflt = rest[9]
    except Exception:
        pass
    nmem = parts[2] if len(parts) > 2 else '?'
    print('pid={} name={} rss={} swap={} majflt={} nvidia_mem={} MiB'.format(
        pid, status.get('Name', '?'), status.get('VmRSS', '?'),
        status.get('VmSwap', '0 kB'), majflt, nmem))
PY
for host in "${HOSTS[@]}"; do
  name=${host%%:*}; ip=${host#*:}
  if [ "$ip" != "127.0.0.1" ]; then
    "${SCP[@]}" /tmp/kw_swap_diag.sh /tmp/kw_gpu_pid_swap.py "$ip:/tmp/" >/dev/null
    "${SSH[@]}" "$ip" 'chmod +x /tmp/kw_swap_diag.sh' >/dev/null || true
  fi
done

cleanup_deployment() {
  docker rm -f glm-dark-head >/dev/null 2>&1 || true
  for ip in 192.168.100.2 192.168.100.3 192.168.100.4; do
    "${SSH[@]}" "$ip" 'docker rm -f glm-dark-worker >/dev/null 2>&1 || true' || true
  done
}

drain_local_swap_cache() {
  sudo -n swapoff -a && sudo -n swapon -a && sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
}

gpu_snapshot() {
  phase=$1
  for host in "${HOSTS[@]}"; do
    name=${host%%:*}; ip=${host#*:}
    {
      echo "===== $phase $name $ip $(date -Is) ====="
      if [ "$ip" = "127.0.0.1" ]; then
        python3 /tmp/kw_gpu_pid_swap.py
        awk '/^(pswpin|pswpout|pgmajfault) /{print}' /proc/vmstat
      else
        "${SSH[@]}" "$ip" 'python3 /tmp/kw_gpu_pid_swap.py; awk "/^(pswpin|pswpout|pgmajfault) /{print}" /proc/vmstat'
      fi
    } | tee -a "$OUTDIR/gpu_snapshots.log"
  done
}

start_samplers() {
  : > "$OUTDIR/sampler_pids.txt"
  for host in "${HOSTS[@]}"; do
    name=${host%%:*}; ip=${host#*:}; remote_out="/tmp/kw_swap_${name}_${STAMP}_${TAG}.log"
    if [ "$ip" = "127.0.0.1" ]; then
      nohup /tmp/kw_swap_diag.sh 10 240 "$remote_out" >/tmp/kw_swap_${name}_${STAMP}_${TAG}.nohup 2>&1 & pid=$!
      echo "$name $ip $pid $remote_out" >> "$OUTDIR/sampler_pids.txt"
    else
      pid=$("${SSH[@]}" "$ip" "nohup /tmp/kw_swap_diag.sh 10 240 '$remote_out' >/tmp/kw_swap_${name}_${STAMP}_${TAG}.nohup 2>&1 & echo \$!")
      echo "$name $ip $pid $remote_out" >> "$OUTDIR/sampler_pids.txt"
    fi
  done
  cat "$OUTDIR/sampler_pids.txt"
}

stop_samplers() {
  while read -r name ip pid remote_out; do
    [ -n "${name:-}" ] || continue
    if [ "$ip" = "127.0.0.1" ]; then
      kill "$pid" >/dev/null 2>&1 || true
      cp "$remote_out" "$OUTDIR/${name}.swap.log" 2>/dev/null || true
    else
      "${SSH[@]}" "$ip" "kill '$pid' >/dev/null 2>&1 || true" || true
      "${SCP[@]}" "$ip:$remote_out" "$OUTDIR/${name}.swap.log" >/dev/null 2>&1 || true
    fi
  done < "$OUTDIR/sampler_pids.txt"
}

wait_ready() {
  local log="/tmp/${SERVED}.log"
  for _ in $(seq 1 120); do
    if docker exec glm-dark-head test -f "$log" >/dev/null 2>&1; then
      if docker exec glm-dark-head grep -q 'Application startup complete' "$log"; then
        echo "READY"
        docker exec glm-dark-head grep -E 'GPU KV cache size|Maximum concurrency|Available KV cache memory|Starting vLLM server|Application startup complete' "$log" | tail -30 | tee "$OUTDIR/readiness.log"
        return 0
      fi
      if docker exec glm-dark-head grep -qE 'EngineCore failed|Traceback|ValueError|CUDA out of memory' "$log"; then
        echo "FAILED"
        docker exec glm-dark-head grep -E 'EngineCore failed|Traceback|ValueError|CUDA out of memory|Free memory|Available KV cache memory|GPU KV cache size|Maximum concurrency' "$log" | tail -120 | tee "$OUTDIR/readiness.log"
        return 1
      fi
    fi
    docker exec glm-dark-head bash -lc 'tail -n 80 /tmp/ray-vllm-head/session_latest/logs/worker-*.err 2>/dev/null | grep -E "Loading safetensors|Loaded weights|Available KV cache|GPU KV cache|Maximum concurrency" | tail -4 || true' | tee -a "$OUTDIR/load_progress.log"
    sleep 30
  done
  echo "TIMEOUT"
  return 2
}

run_decode_probe() {
  python3 - "$SERVED" <<'PY' | tee "$OUTDIR/decode_probe.log"
import json, sys, time, urllib.request
model = sys.argv[1]
url = 'http://192.168.100.1:18089/v1/completions'
def call(label, prompt, max_tokens):
    payload = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': 0.0, 'top_p': 1.0}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as resp:
        body = json.loads(resp.read().decode())
    dt = time.perf_counter() - t0
    toks = (body.get('usage') or {}).get('completion_tokens') or 0
    print(f'{label}: tokens={toks} seconds={dt:.3f} tps={(toks/dt if dt else 0):.3f}')
    txt = (body.get('choices') or [{}])[0].get('text','')
    print('  prefix=', repr(txt[:160].replace('\n',' ')))
warm = 'Write me code to efficiently print prime numbers from 0 to 1,000,000,000. Explain tradeoffs briefly.'
probe = 'Write me a compact but complete FastAPI + Pydantic + SQLite + SQLAlchemy + React kanban app. Include backend and frontend code.'
call('warmup_primes_128', warm, 128)
print('MEASURED_DECODE_BEGIN')
call('measured_codegen_512', probe, 512)
print('MEASURED_DECODE_END')
PY
}

analyze() {
  python3 - "$OUTDIR" <<'PY' | tee "$OUTDIR/analysis.txt"
import re, sys
from pathlib import Path
out = Path(sys.argv[1])
print('analysis_dir=', out)
# Direct before/after GPU worker deltas.
snap = (out / 'gpu_snapshots.log').read_text(errors='replace') if (out / 'gpu_snapshots.log').exists() else ''
blocks = re.split(r'^===== ', snap, flags=re.M)
records = {}
for b in blocks:
    if not b.strip():
        continue
    header, *lines = b.splitlines()
    parts = header.split()
    if len(parts) < 3:
        continue
    phase, host = parts[0], parts[1]
    rec = records.setdefault(host, {})
    data = {'gpu': [], 'vm': {}}
    for line in lines:
        m = re.match(r'pid=(\d+) name=(\S+) rss=(\d+) kB swap=(\d+) kB majflt=(\d+) nvidia_mem=(\d+) MiB', line)
        if m:
            data['gpu'].append({
                'pid': int(m.group(1)), 'name': m.group(2), 'rss_kb': int(m.group(3)),
                'swap_kb': int(m.group(4)), 'majflt': int(m.group(5)), 'nvidia_mib': int(m.group(6)),
            })
        m = re.match(r'(pswpin|pswpout|pgmajfault)\s+(\d+)', line)
        if m:
            data['vm'][m.group(1)] = int(m.group(2))
    rec[phase] = data
for host, rec in sorted(records.items()):
    before = rec.get('BEFORE_MEASURED_DECODE')
    after = rec.get('AFTER_MEASURED_DECODE')
    if not before or not after:
        continue
    print('\nHOST', host)
    bg = before['gpu'][0] if before['gpu'] else None
    ag = after['gpu'][0] if after['gpu'] else None
    if bg and ag:
        print('gpu_swap_kb_before=', bg['swap_kb'], 'after=', ag['swap_kb'], 'delta=', ag['swap_kb'] - bg['swap_kb'])
        print('gpu_majflt_before=', bg['majflt'], 'after=', ag['majflt'], 'delta=', ag['majflt'] - bg['majflt'])
        print('gpu_rss_kb_before=', bg['rss_kb'], 'after=', ag['rss_kb'], 'delta=', ag['rss_kb'] - bg['rss_kb'])
    for k in ['pswpin', 'pswpout', 'pgmajfault']:
        if k in before['vm'] and k in after['vm']:
            print(k + '_delta=', after['vm'][k] - before['vm'][k])
# Sampler summary.
for path in sorted(out.glob('*.swap.log')):
    totals = {}
    maxdelta = {}
    gpu_lines = 0
    for line in path.read_text(errors='replace').splitlines():
        m = re.match(r'(pswpin|pswpout|pgmajfault|pgfault)\s+\d+\s+delta=(-?\d+)', line)
        if m:
            k, d = m.group(1), int(m.group(2))
            totals[k] = totals.get(k, 0) + d
            maxdelta[k] = max(maxdelta.get(k, 0), d)
        if line.startswith('pid=') and 'nvidia_mem=' in line:
            gpu_lines += 1
    print('\nSAMPLER', path.name)
    print('gpu_pid_samples=', gpu_lines)
    for k in ['pswpin', 'pswpout', 'pgmajfault']:
        print(k, 'total_delta=', totals.get(k, 0), 'max_sample_delta=', maxdelta.get(k, 0))
PY
}

cleanup_deployment
start_samplers
drain_local_swap_cache
ENV_FILE="$PWD/$ENV_FILE" PATCH_DIAGNOSTICS=1 ./launch-glm52-mtp3-dcp4-128k.sh | tee "$OUTDIR/launch.log"
if wait_ready; then
  gpu_snapshot BEFORE_WARMUP
  # Warmup is included in run_decode_probe; use a separate before snapshot immediately before measured decode by splitting with one direct warmup first.
  python3 - "$SERVED" <<'PY' | tee "$OUTDIR/warmup.log"
import json, sys, time, urllib.request
model = sys.argv[1]
url = 'http://192.168.100.1:18089/v1/completions'
payload = {'model': model, 'prompt': 'Write me code to efficiently print prime numbers from 0 to 1,000,000,000. Explain tradeoffs briefly.', 'max_tokens': 128, 'temperature': 0.0, 'top_p': 1.0}
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=900) as resp:
    body = json.loads(resp.read().decode())
dt = time.perf_counter() - t0
toks = (body.get('usage') or {}).get('completion_tokens') or 0
print('warmup_primes_128: tokens={} seconds={:.3f} tps={:.3f}'.format(toks, dt, toks/dt if dt else 0))
PY
  gpu_snapshot BEFORE_MEASURED_DECODE
  python3 - "$SERVED" <<'PY' | tee "$OUTDIR/decode_probe.log"
import json, sys, time, urllib.request
model = sys.argv[1]
url = 'http://192.168.100.1:18089/v1/completions'
payload = {'model': model, 'prompt': 'Write me a compact but complete FastAPI + Pydantic + SQLite + SQLAlchemy + React kanban app. Include backend and frontend code.', 'max_tokens': 512, 'temperature': 0.0, 'top_p': 1.0}
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=900) as resp:
    body = json.loads(resp.read().decode())
dt = time.perf_counter() - t0
toks = (body.get('usage') or {}).get('completion_tokens') or 0
print('measured_codegen_512: tokens={} seconds={:.3f} tps={:.3f}'.format(toks, dt, toks/dt if dt else 0))
print('prefix=', repr(((body.get('choices') or [{}])[0].get('text',''))[:160].replace('\n',' ')))
PY
  gpu_snapshot AFTER_MEASURED_DECODE
  docker exec glm-dark-head bash -lc "cat /tmp/${SERVED}.log 2>/dev/null || true" > "$OUTDIR/vllm.log" || true
fi
cleanup_deployment
stop_samplers
drain_local_swap_cache
analyze

echo "OUTDIR=$OUTDIR"
