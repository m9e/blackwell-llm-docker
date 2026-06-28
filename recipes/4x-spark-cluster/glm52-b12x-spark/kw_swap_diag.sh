#!/usr/bin/env bash
set -euo pipefail
interval=${1:-5}
samples=${2:-12}
out=${3:-/tmp/kw_swap_diag_$(hostname)_$(date +%Y%m%d-%H%M%S).log}
read_vm() {
  awk '/^(pswpin|pswpout|pgmajfault|pgfault|pgscan_kswapd|pgscan_direct|pgsteal_kswapd|pgsteal_direct|oom_kill) /{print}' /proc/vmstat
}
proc_swap() {
  for p in /proc/[0-9]*; do
    pid=${p##*/}
    [ -r "$p/status" ] || continue
    name=$(awk '/^Name:/{print $2; exit}' "$p/status" 2>/dev/null || true)
    swap=$(awk '/^VmSwap:/{print $2; exit}' "$p/status" 2>/dev/null || echo 0)
    rss=$(awk '/^VmRSS:/{print $2; exit}' "$p/status" 2>/dev/null || echo 0)
    [ "${swap:-0}" -gt 0 ] 2>/dev/null && printf '%10s kB swap %10s kB rss pid=%s name=%s\n' "$swap" "${rss:-0}" "$pid" "$name"
  done | sort -nr | sed -n '1,30p'
}
gpu_pids() {
  python3 - <<'PY'
import subprocess

try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    out = ""

if not out.strip():
    print("no nvidia compute pids")

for line in out.splitlines():
    parts = [x.strip() for x in line.split(",")]
    if not parts or not parts[0].isdigit():
        continue
    pid = parts[0]
    status = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for item in f:
                if item.startswith(("Name:", "VmRSS:", "VmSwap:")):
                    key, value = item.split(":", 1)
                    status[key] = value.strip()
    except FileNotFoundError:
        continue
    majflt = "?"
    try:
        stat = open(f"/proc/{pid}/stat").read()
        rest = stat.rsplit(") ", 1)[1].split()
        majflt = rest[9]
    except Exception:
        pass
    nvidia_mem = parts[2] if len(parts) > 2 else "?"
    print(
        "pid={} name={} rss={} swap={} majflt={} nvidia_mem={} MiB".format(
            pid,
            status.get("Name", "?"),
            status.get("VmRSS", "?"),
            status.get("VmSwap", "0 kB"),
            majflt,
            nvidia_mem,
        )
    )
PY
}
{
  echo "# kw swap diag host=$(hostname) start=$(date -Is) interval=${interval}s samples=${samples}"
  prev=$(mktemp)
  cur=$(mktemp)
  read_vm > "$prev"
  for i in $(seq 1 "$samples"); do
    sleep "$interval"
    echo "===== sample=$i time=$(date -Is) ====="
    read_vm > "$cur"
    echo "-- vmstat si/so --"
    vmstat 1 2 | tail -1
    echo "-- vmstat deltas --"
    awk 'NR==FNR{a[$1]=$2; next} {printf "%s %s delta=%s\n", $1, $2, ($2-(a[$1]+0))}' "$prev" "$cur"
    echo "-- meminfo --"
    awk '/^(SwapTotal|SwapFree|SwapCached|MemFree|MemAvailable):/{print}' /proc/meminfo
    echo "-- top process swap --"
    proc_swap
    echo "-- nvidia compute pids --"
    gpu_pids
    mv "$cur" "$prev"
    cur=$(mktemp)
  done
  rm -f "$prev" "$cur"
  echo "# end=$(date -Is)"
} | tee "$out"
echo "wrote $out" >&2
