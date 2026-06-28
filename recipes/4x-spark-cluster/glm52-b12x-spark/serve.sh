#!/usr/bin/env bash
set -euo pipefail

# Start GLM-5.2 serving inside an already-running Spark Ray cluster.
# Run launch-ray.sh first. This wrapper keeps the Spark-specific choices
# explicit and avoids the single-host RTX defaults in upstream serve-glm52.sh.

HEAD_NAME="${HEAD_NAME:-glm-dark-head}"
HEAD_IP="${HEAD_IP:-192.168.100.1}"
RAY_PORT="${RAY_PORT:-26479}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18089}"
PROFILE="${PROFILE:-bf16-dcp4-48k-no-mtp}"
HS_IFACE="${HS_IFACE:-enP2p1s0f0np0}"
STOP_EXISTING_API="${STOP_EXISTING_API:-1}"

MODEL="/models"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
DCP_SIZE="${DCP_SIZE:-}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
DCP_KV_CACHE_INTERLEAVE_SIZE="${DCP_KV_CACHE_INTERLEAVE_SIZE:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
USE_LOCAL_ARGMAX_REDUCTION="${USE_LOCAL_ARGMAX_REDUCTION:-0}"
REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD:-standard}"
SYNTHETIC_ACCEPTANCE_RATES="${SYNTHETIC_ACCEPTANCE_RATES:-}"
SYNTHETIC_ACCEPTANCE_LENGTH="${SYNTHETIC_ACCEPTANCE_LENGTH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
ENABLE_MTP="${ENABLE_MTP:-}"
LOG_FILE="${LOG_FILE:-/tmp/glm52-spark-${PROFILE}.log}"
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
KZ_KV_DIAG="${KZ_KV_DIAG:-0}"
VLLM_KZ_TRIM_AFTER_LOAD="${VLLM_KZ_TRIM_AFTER_LOAD:-0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"
DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}"
USE_B12X_SPARSE_INDEXER="${USE_B12X_SPARSE_INDEXER:-1}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
REASONING_PARSER="${REASONING_PARSER:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
B12X_W4A16_TC_DECODE="${B12X_W4A16_TC_DECODE:-1}"
B12X_DENSE_SPLITK_TURBO="${B12X_DENSE_SPLITK_TURBO:-1}"
VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK:-1}"
VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT:-1}"
VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP="${VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP:-}"
VLLM_MTP_BROADCAST_DRAFT_TOKENS="${VLLM_MTP_BROADCAST_DRAFT_TOKENS:-0}"
VLLM_MTP_DCP_DIAG="${VLLM_MTP_DCP_DIAG:-0}"
VLLM_MTP_DCP_DIAG_LIMIT="${VLLM_MTP_DCP_DIAG_LIMIT:-8}"
VLLM_MTP_DCP_DIAG_TOKENS="${VLLM_MTP_DCP_DIAG_TOKENS:-8}"
VLLM_B12X_MLA_EXACT_SEQ_LENS="${VLLM_B12X_MLA_EXACT_SEQ_LENS:-0}"
RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

INDEX_TOPK_PATTERN="${INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
if [[ -z "${HF_OVERRIDES:-}" ]]; then
  HF_OVERRIDES="{\"use_index_cache\":true,\"index_topk_pattern\":\"${INDEX_TOPK_PATTERN}\"}"
fi

case "${PROFILE}" in
  bf16-dcp4-48k-no-mtp)
    DCP_SIZE="${DCP_SIZE:-4}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.917}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    ENABLE_MTP="${ENABLE_MTP:-0}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm52-spark-b12x-bf16-dcp4-48k}"
    ;;
  fp8-dcp2-28k-mtp3)
    DCP_SIZE="${DCP_SIZE:-2}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-28672}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.918}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_ds_mla}"
    ENABLE_MTP="${ENABLE_MTP:-1}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm52-spark-b12x-fp8-dcp2-28k-mtp3}"
    ;;
  fp8-dcp4-16k-mtp3)
    DCP_SIZE="${DCP_SIZE:-4}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.918}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_ds_mla}"
    ENABLE_MTP="${ENABLE_MTP:-1}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm52-spark-b12x-fp8-dcp4-16k-mtp3}"
    ;;
  custom)
    : "${DCP_SIZE:?DCP_SIZE is required for PROFILE=custom}"
    : "${MAX_MODEL_LEN:?MAX_MODEL_LEN is required for PROFILE=custom}"
    : "${MAX_NUM_BATCHED_TOKENS:?MAX_NUM_BATCHED_TOKENS is required for PROFILE=custom}"
    : "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION is required for PROFILE=custom}"
    : "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required for PROFILE=custom}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    ENABLE_MTP="${ENABLE_MTP:-0}"
    ;;
  *)
    echo "Unknown PROFILE '${PROFILE}'" >&2
    echo "Known profiles: bf16-dcp4-48k-no-mtp, fp8-dcp2-28k-mtp3, fp8-dcp4-16k-mtp3, custom" >&2
    exit 2
    ;;
esac

docker exec "${HEAD_NAME}" bash -lc "ray status --address=${HEAD_IP}:${RAY_PORT} | grep -q '/4.0 GPU'"

if [[ "${STOP_EXISTING_API}" == "1" ]]; then
  docker exec "${HEAD_NAME}" bash -lc "pkill -f '[v]llm.entrypoints.openai.api_server' >/dev/null 2>&1 || true"
  sleep 2
fi

docker_env=()
if [[ -n "${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS}" ]]; then
  docker_env+=(
    -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS}"
  )
fi

docker exec -d \
  "${docker_env[@]}" \
  -e SAFETENSORS_FAST_GPU=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e NCCL_SOCKET_IFNAME="${HS_IFACE}" \
  -e GLOO_SOCKET_IFNAME="${HS_IFACE}" \
  -e NCCL_IB_DISABLE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM}" \
  -e VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE}" \
  -e VLLM_DISABLE_TP_MQ_BROADCASTER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_USE_B12X_SPARSE_INDEXER="${USE_B12X_SPARSE_INDEXER}" \
  -e VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM}" \
  -e VLLM_DCP_GLOBAL_TOPK="${VLLM_DCP_GLOBAL_TOPK}" \
  -e VLLM_DCP_SHARD_DRAFT="${VLLM_DCP_SHARD_DRAFT}" \
  -e VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP="${VLLM_MTP_RECOMPUTE_TOPK_FROM_STEP}" \
  -e VLLM_MTP_BROADCAST_DRAFT_TOKENS="${VLLM_MTP_BROADCAST_DRAFT_TOKENS}" \
  -e VLLM_DCP_SYNC_REJECTION_OUTPUT="${VLLM_DCP_SYNC_REJECTION_OUTPUT:-0}" \
  -e VLLM_MTP_DCP_DIAG="${VLLM_MTP_DCP_DIAG}" \
  -e VLLM_MTP_DCP_DIAG_LIMIT="${VLLM_MTP_DCP_DIAG_LIMIT}" \
  -e VLLM_MTP_DCP_DIAG_TOKENS="${VLLM_MTP_DCP_DIAG_TOKENS}" \
      -e VLLM_B12X_MLA_EXACT_SEQ_LENS="${VLLM_B12X_MLA_EXACT_SEQ_LENS}" \
      -e VLLM_MTP_DRAFT_PROB_DIAG="${VLLM_MTP_DRAFT_PROB_DIAG}" \
      -e VLLM_MTP_DRAFT_PROB_DIAG_TOPK="${VLLM_MTP_DRAFT_PROB_DIAG_TOPK}" \
  -e RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS}" \
  -e KZ_KV_DIAG="${KZ_KV_DIAG}" \
  -e VLLM_KZ_TRIM_AFTER_LOAD="${VLLM_KZ_TRIM_AFTER_LOAD}" \
  -e USES_B12X=True \
  -e B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16}" \
  -e B12X_W4A16_TC_DECODE="${B12X_W4A16_TC_DECODE}" \
  -e B12X_DENSE_SPLITK_TURBO="${B12X_DENSE_SPLITK_TURBO}" \
  -e RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
  "${HEAD_NAME}" bash -lc "$(cat <<EOF
set -euo pipefail
args=(
  python3 -m vllm.entrypoints.openai.api_server
  --model '${MODEL}'
  --tokenizer '${MODEL}'
  --served-model-name '${SERVED_MODEL_NAME}'
  --trust-remote-code
  --download-dir '${MODEL}'
  --load-format '${LOAD_FORMAT}'
  --quantization '${QUANTIZATION}'
  --distributed-executor-backend ray
  --tensor-parallel-size '${TP_SIZE}'
  --decode-context-parallel-size '${DCP_SIZE}'
  --dcp-comm-backend '${DCP_COMM_BACKEND}'
  --dcp-kv-cache-interleave-size '${DCP_KV_CACHE_INTERLEAVE_SIZE}'
  --pipeline-parallel-size '${PP_SIZE}'
  --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}'
  --max-model-len '${MAX_MODEL_LEN}'
  --max-num-seqs '${MAX_NUM_SEQS}'
  --max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}'
  --generation-config vllm
  --hf-overrides '${HF_OVERRIDES}'
  --port '${PORT}'
  --host '${HOST}'
  --no-enable-log-requests
)
if [[ '${ENABLE_PREFIX_CACHING}' == '0' ]]; then
  args+=(--no-enable-prefix-caching)
fi
if [[ '${DISABLE_ASYNC_SCHEDULING:-0}' == '1' ]]; then
  args+=(--no-async-scheduling)
fi
if [[ '${ENFORCE_EAGER}' == '1' ]]; then
  args+=(--enforce-eager)
fi
if [[ -n '${MAX_CUDAGRAPH_CAPTURE_SIZE}' ]]; then
  args+=(--max-cudagraph-capture-size '${MAX_CUDAGRAPH_CAPTURE_SIZE}')
fi
if [[ -n '${KV_CACHE_MEMORY_BYTES}' ]]; then
  args+=(--kv-cache-memory-bytes '${KV_CACHE_MEMORY_BYTES}')
fi
if [[ '${KV_CACHE_DTYPE}' != 'auto' ]]; then
  args+=(--kv-cache-dtype '${KV_CACHE_DTYPE}')
fi
if [[ '${ATTENTION_BACKEND}' != 'auto' ]]; then
  args+=(--attention-backend '${ATTENTION_BACKEND}')
fi
if [[ '${MOE_BACKEND}' != 'auto' ]]; then
  args+=(--moe-backend '${MOE_BACKEND}')
fi
if [[ -n '${REASONING_PARSER}' ]]; then
  args+=(--reasoning-parser '${REASONING_PARSER}')
fi
if [[ -n '${TOOL_CALL_PARSER}' ]]; then
  args+=(--tool-call-parser '${TOOL_CALL_PARSER}')
fi
if [[ '${ENABLE_AUTO_TOOL_CHOICE}' == '1' ]]; then
  args+=(--enable-auto-tool-choice)
fi
if [[ '${ENABLE_MTP}' == '1' ]]; then
  speculative_config='{"model":"${MODEL}","method":"mtp","num_speculative_tokens":${NUM_SPECULATIVE_TOKENS},"moe_backend":"${MOE_BACKEND}","draft_attention_backend":"${DRAFT_ATTENTION_BACKEND}","draft_sample_method":"probabilistic"'
  if [[ '${USE_LOCAL_ARGMAX_REDUCTION}' == '1' ]]; then
    speculative_config+=',"use_local_argmax_reduction":true'
  fi
  if [[ '${REJECTION_SAMPLE_METHOD}' != 'standard' ]]; then
    speculative_config+=',"rejection_sample_method":"${REJECTION_SAMPLE_METHOD}"'
  fi
  if [[ -n '${SYNTHETIC_ACCEPTANCE_RATES}' ]]; then
    speculative_config+=',"synthetic_acceptance_rates":[${SYNTHETIC_ACCEPTANCE_RATES}]'
  fi
  if [[ -n '${SYNTHETIC_ACCEPTANCE_LENGTH}' ]]; then
    speculative_config+=',"synthetic_acceptance_length":${SYNTHETIC_ACCEPTANCE_LENGTH}'
  fi
  speculative_config+='}'
  args+=(--speculative-config "\${speculative_config}")
fi
printf 'Starting %s on port %s\\n' '${SERVED_MODEL_NAME}' '${PORT}' >'${LOG_FILE}'
printf '%q ' "\${args[@]}" >>'${LOG_FILE}'
printf '\\n' >>'${LOG_FILE}'
exec "\${args[@]}" >>'${LOG_FILE}' 2>&1
EOF
)"

echo "Started ${SERVED_MODEL_NAME}; log: docker exec ${HEAD_NAME} tail -f ${LOG_FILE}"
echo "Endpoint: http://${HEAD_IP}:${PORT}/v1"
