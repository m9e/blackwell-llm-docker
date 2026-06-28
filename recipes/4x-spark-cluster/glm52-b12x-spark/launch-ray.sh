#!/usr/bin/env bash
set -euo pipefail

# Spark-specific 4-node Ray launcher for the GLM-5.2 B12X vLLM stack.
# This intentionally avoids the upstream RTX/PCIe defaults:
# - one GB10 GPU per host
# - RoCE/NCCL over the Spark high-speed interface
# - tiny Ray object store
# - no B12X PCIe allreduce

IMAGE="${IMAGE:-glm-darkdevotion-b12x:20260625-arm64-mtp1-trim}"
MODEL_DIR="${MODEL_DIR:-/var/tmp/models/Mapika/GLM-5.2-NVFP4-MTP-hybrid}"
HEAD_IP="${HEAD_IP:-192.168.100.1}"
RAY_PORT="${RAY_PORT:-26479}"
OBJECT_STORE="${OBJECT_STORE:-134217728}"
OBJECT_SPILLING_DIR="${OBJECT_SPILLING_DIR:-/var/tmp/ray-spill}"
WORKER_IPS="${WORKER_IPS:-192.168.100.2 192.168.100.3 192.168.100.4}"
SSH_KEY="${SSH_KEY:-/etc/kamiwaza/ssl/cluster.key}"
HS_IFACE="${HS_IFACE:-enP2p1s0f0np0}"
HEAD_NAME="${HEAD_NAME:-glm-dark-head}"
WORKER_NAME="${WORKER_NAME:-glm-dark-worker}"
DROP_CACHES="${DROP_CACHES:-1}"
VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
KZ_KV_DIAG="${KZ_KV_DIAG:-0}"
VLLM_KZ_TRIM_AFTER_LOAD="${VLLM_KZ_TRIM_AFTER_LOAD:-0}"
VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}"
VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-1}"
VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP="${VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP:-}"
VLLM_MTP_BROADCAST_DRAFT_TOKENS="${VLLM_MTP_BROADCAST_DRAFT_TOKENS:-0}"
B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
B12X_W4A16_TC_DECODE="${B12X_W4A16_TC_DECODE:-1}"
B12X_DENSE_SPLITK_TURBO="${B12X_DENSE_SPLITK_TURBO:-1}"

ssh_base=(
  ssh
  -i "${SSH_KEY}"
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
  -o BatchMode=yes
  -o ConnectTimeout=10
)

docker_common=(
  --network host
  --ipc host
  --privileged
  --security-opt label=disable
  --gpus all
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "${MODEL_DIR}:/models:ro"
  -e RAY_memory_usage_threshold=0.99
  -e RAY_memory_monitor_refresh_ms=0
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e CUDA_DEVICE_MAX_CONNECTIONS=32
  -e NCCL_SOCKET_IFNAME="${HS_IFACE}"
  -e GLOO_SOCKET_IFNAME="${HS_IFACE}"
  -e NCCL_IB_DISABLE=0
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e SAFETENSORS_FAST_GPU=1
  -e CUTE_DSL_ARCH=sm_121a
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e VLLM_USE_FLASHINFER_SAMPLER=1
  -e VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM}"
  -e VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE}"
  -e VLLM_DISABLE_TP_MQ_BROADCASTER=1
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0
  -e VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER}"
  -e VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM}"
  -e VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK}"
  -e VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT}"
  -e VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP="${VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP}"
  -e VLLM_MTP_BROADCAST_DRAFT_TOKENS="${VLLM_MTP_BROADCAST_DRAFT_TOKENS}"
  -e VLLM_DCP_SYNC_REJECTION_OUTPUT="${VLLM_DCP_SYNC_REJECTION_OUTPUT:-0}"
  -e KZ_KV_DIAG="${KZ_KV_DIAG}"
  -e VLLM_KZ_TRIM_AFTER_LOAD="${VLLM_KZ_TRIM_AFTER_LOAD}"
  -e USES_B12X=True
  -e B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16}"
  -e B12X_W4A16_TC_DECODE="${B12X_W4A16_TC_DECODE}"
  -e B12X_DENSE_SPLITK_TURBO="${B12X_DENSE_SPLITK_TURBO}"
)

stop_all() {
  docker rm -f "${HEAD_NAME}" >/dev/null 2>&1 || true
  for ip in ${WORKER_IPS}; do
    "${ssh_base[@]}" "${ip}" "docker rm -f '${WORKER_NAME}' >/dev/null 2>&1 || true" &
  done
  wait
}

drop_caches_all() {
  if [[ "${DROP_CACHES}" != "1" ]]; then
    return 0
  fi
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
  for ip in ${WORKER_IPS}; do
    "${ssh_base[@]}" "${ip}" "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true" &
  done
  wait
}

start_head() {
  docker run -d --name "${HEAD_NAME}" \
    "${docker_common[@]}" \
    -e VLLM_HOST_IP="${HEAD_IP}" \
    -e HOST_IP="${HEAD_IP}" \
    "${IMAGE}" \
    bash -lc "mkdir -p '${OBJECT_SPILLING_DIR}' && ray start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --object-store-memory=${OBJECT_STORE} --object-spilling-directory='${OBJECT_SPILLING_DIR}' --num-cpus=1 --num-gpus=1 --include-dashboard=false --include-log-monitor=false --disable-usage-stats --temp-dir=/tmp/ray-vllm-head --block" \
    >/tmp/glm-dark-head.cid
}

start_workers() {
  for ip in ${WORKER_IPS}; do
    "${ssh_base[@]}" "${ip}" \
      "docker run -d --name '${WORKER_NAME}' --network host --ipc host --privileged --security-opt label=disable --gpus all --ulimit memlock=-1 --ulimit stack=67108864 -v '${MODEL_DIR}:/models:ro' -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e NCCL_SOCKET_IFNAME='${HS_IFACE}' -e GLOO_SOCKET_IFNAME='${HS_IFACE}' -e NCCL_IB_DISABLE=0 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 -e CUTE_DSL_ARCH=sm_121a -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_B12X_FP8_GEMM='${VLLM_USE_B12X_FP8_GEMM}' -e VLLM_USE_B12X_MOE='${VLLM_USE_B12X_MOE}' -e VLLM_DISABLE_TP_MQ_BROADCASTER=1 -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e VLLM_USE_B12X_SPARSE_INDEXER='${VLLM_USE_B12X_SPARSE_INDEXER}' -e VLLM_USE_DEEP_GEMM='${VLLM_USE_DEEP_GEMM}' -e VLLM_DCP_GLOBAL_TOPK='${VLLM_DCP_GLOBAL_TOPK}' -e VLLM_DCP_SHARD_DRAFT='${VLLM_DCP_SHARD_DRAFT}' -e VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP='${VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP}' -e VLLM_MTP_BROADCAST_DRAFT_TOKENS='${VLLM_MTP_BROADCAST_DRAFT_TOKENS}' -e VLLM_DCP_SYNC_REJECTION_OUTPUT='${VLLM_DCP_SYNC_REJECTION_OUTPUT:-0}' -e KZ_KV_DIAG='${KZ_KV_DIAG}' -e VLLM_KZ_TRIM_AFTER_LOAD='${VLLM_KZ_TRIM_AFTER_LOAD}' -e USES_B12X=True -e B12X_MOE_FORCE_A16='${B12X_MOE_FORCE_A16}' -e B12X_W4A16_TC_DECODE='${B12X_W4A16_TC_DECODE}' -e B12X_DENSE_SPLITK_TURBO='${B12X_DENSE_SPLITK_TURBO}' -e VLLM_HOST_IP='${ip}' -e HOST_IP='${ip}' '${IMAGE}' bash -lc \"mkdir -p '${OBJECT_SPILLING_DIR}' && ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${ip} --object-store-memory=${OBJECT_STORE} --object-spilling-directory='${OBJECT_SPILLING_DIR}' --num-cpus=1 --num-gpus=1 --include-log-monitor=false --disable-usage-stats --temp-dir=/tmp/ray-vllm-worker --block\" >/tmp/glm-dark-worker.cid" &
  done
  wait
}

wait_cluster() {
  for _ in $(seq 1 60); do
    if docker exec "${HEAD_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} 2>/dev/null | grep -q '/4.0 GPU'"; then
      docker exec "${HEAD_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} | sed -n '1,80p'"
      return 0
    fi
    sleep 3
  done
  docker exec "${HEAD_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} || true"
  return 1
}

stop_all
drop_caches_all
start_head
sleep 5
start_workers
wait_cluster
