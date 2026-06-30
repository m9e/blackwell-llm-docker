#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKER_IPS="${WORKER_IPS:-192.168.100.2 192.168.100.3 192.168.100.4}"
SSH_KEY="${SSH_KEY:-/etc/kamiwaza/ssl/cluster.key}"
HEAD_NAME="${HEAD_NAME:-glm-dark-head}"
WORKER_NAME="${WORKER_NAME:-glm-dark-worker}"
DRAIN_SWAP="${DRAIN_SWAP:-0}"
DROP_CACHES="${DROP_CACHES:-0}"

ssh_base=(
  ssh
  -i "${SSH_KEY}"
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
  -o BatchMode=yes
  -o ConnectTimeout=10
)

stop_head() {
  docker rm -f "${HEAD_NAME}" >/dev/null 2>&1 || true
}

stop_workers() {
  for ip in ${WORKER_IPS}; do
    "${ssh_base[@]}" "${ip}" "docker rm -f '${WORKER_NAME}' >/dev/null 2>&1 || true" &
  done
  wait
}

clean_memory_head() {
  if [[ "${DRAIN_SWAP}" == "1" ]]; then
    sudo -n swapoff -a && sudo -n swapon -a || true
  fi
  if [[ "${DROP_CACHES}" == "1" ]]; then
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
  fi
}

clean_memory_workers() {
  if [[ "${DRAIN_SWAP}" != "1" && "${DROP_CACHES}" != "1" ]]; then
    return 0
  fi
  for ip in ${WORKER_IPS}; do
    cmd="true"
    if [[ "${DRAIN_SWAP}" == "1" ]]; then
      cmd="${cmd}; sudo -n swapoff -a && sudo -n swapon -a || true"
    fi
    if [[ "${DROP_CACHES}" == "1" ]]; then
      cmd="${cmd}; sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true"
    fi
    "${ssh_base[@]}" "${ip}" "${cmd}" &
  done
  wait
}

stop_head
stop_workers
clean_memory_head
clean_memory_workers

echo "Stopped GLM-5.2 Spark containers: ${HEAD_NAME}, ${WORKER_NAME}"
if [[ "${DRAIN_SWAP}" == "1" || "${DROP_CACHES}" == "1" ]]; then
  echo "Memory cleanup requested: DRAIN_SWAP=${DRAIN_SWAP} DROP_CACHES=${DROP_CACHES}"
fi
