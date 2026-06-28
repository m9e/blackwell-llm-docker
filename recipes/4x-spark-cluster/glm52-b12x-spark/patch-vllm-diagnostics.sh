#!/usr/bin/env bash
set -euo pipefail

HEAD_NAME="${HEAD_NAME:-glm-dark-head}"
WORKER_NAME="${WORKER_NAME:-glm-dark-worker}"
WORKER_IPS="${WORKER_IPS:-192.168.100.2 192.168.100.3 192.168.100.4}"
SSH_KEY="${SSH_KEY:-/etc/kamiwaza/ssl/cluster.key}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-/home/matt/code/vllm-dark-devotion}"

archive="$(mktemp /tmp/kz-vllm-diag.XXXXXX.tar)"
cleanup() {
  rm -f "${archive}"
}
trap cleanup EXIT

tar -C "${VLLM_SOURCE_DIR}" -cf "${archive}" \
  vllm/v1/core/kv_cache_utils.py \
  vllm/v1/worker/gpu_worker.py \
  vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
  vllm/v1/sample/rejection_sampler.py

install_in_container() {
  local docker_prefix=("$@")
  local parent
  parent="$("${docker_prefix[@]}" exec "${HEAD_NAME}" python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')"
  "${docker_prefix[@]}" cp "${archive}" "${HEAD_NAME}:/tmp/kz-vllm-diag.tar"
  "${docker_prefix[@]}" exec "${HEAD_NAME}" bash -lc \
    "tar -xf /tmp/kz-vllm-diag.tar -C '${parent}' && rm -f /tmp/kz-vllm-diag.tar"
}

install_head() {
  local parent
  parent="$(docker exec "${HEAD_NAME}" python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')"
  docker cp "${archive}" "${HEAD_NAME}:/tmp/kz-vllm-diag.tar"
  docker exec "${HEAD_NAME}" bash -lc \
    "tar -xf /tmp/kz-vllm-diag.tar -C '${parent}' && rm -f /tmp/kz-vllm-diag.tar"
}

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

install_head

for ip in ${WORKER_IPS}; do
  remote_archive="/tmp/kz-vllm-diag-${USER:-matt}.tar"
  "${scp_base[@]}" "${archive}" "${ip}:${remote_archive}"
  "${ssh_base[@]}" "${ip}" "set -euo pipefail
parent=\$(docker exec '${WORKER_NAME}' python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')
docker cp '${remote_archive}' '${WORKER_NAME}:/tmp/kz-vllm-diag.tar'
docker exec '${WORKER_NAME}' bash -lc \"tar -xf /tmp/kz-vllm-diag.tar -C '\${parent}' && rm -f /tmp/kz-vllm-diag.tar\"
rm -f '${remote_archive}'" &
done
wait

echo "Patched vLLM diagnostics into ${HEAD_NAME} and workers: ${WORKER_IPS}"
