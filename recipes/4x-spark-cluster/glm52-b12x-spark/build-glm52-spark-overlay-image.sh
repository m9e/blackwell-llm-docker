#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_IMAGE="${BASE_IMAGE:-glm-darkdevotion-b12x:20260624-arm64-mtpfix6}"
IMAGE="${IMAGE:-glm-darkdevotion-b12x:20260625-arm64-mtp1-trim}"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-/home/matt/code/vllm-dark-devotion}"
BUILD_WORKERS="${BUILD_WORKERS:-1}"
WORKER_IPS="${WORKER_IPS:-192.168.100.2 192.168.100.3 192.168.100.4}"
SSH_KEY="${SSH_KEY:-/etc/kamiwaza/ssl/cluster.key}"

tmpdir="$(mktemp -d /tmp/glm52-spark-overlay.XXXXXX)"
cleanup() {
  rm -rf "${tmpdir}"
}
trap cleanup EXIT

mkdir -p \
  "${tmpdir}/vllm/v1/core" \
  "${tmpdir}/vllm/v1/attention/backends/mla" \
  "${tmpdir}/vllm/v1/sample" \
  "${tmpdir}/vllm/v1/spec_decode" \
  "${tmpdir}/vllm/v1/worker" \
  "${tmpdir}/vllm/v1/worker/gpu/spec_decode" \
  "${tmpdir}/vllm/model_executor/models"

cp "${VLLM_SOURCE_DIR}/vllm/v1/core/kv_cache_utils.py" \
  "${tmpdir}/vllm/v1/core/kv_cache_utils.py"
cp "${VLLM_SOURCE_DIR}/vllm/v1/attention/backends/mla/indexer.py" \
  "${tmpdir}/vllm/v1/attention/backends/mla/indexer.py"
cp "${VLLM_SOURCE_DIR}/vllm/v1/spec_decode/step3p5.py" \
  "${tmpdir}/vllm/v1/spec_decode/step3p5.py"
cp "${VLLM_SOURCE_DIR}/vllm/v1/spec_decode/llm_base_proposer.py" \
  "${tmpdir}/vllm/v1/spec_decode/llm_base_proposer.py"
cp "${VLLM_SOURCE_DIR}/vllm/v1/sample/rejection_sampler.py" \
  "${tmpdir}/vllm/v1/sample/rejection_sampler.py"
cp "${VLLM_SOURCE_DIR}/vllm/v1/worker/gpu_worker.py" \
  "${tmpdir}/vllm/v1/worker/gpu_worker.py"
cp "${VLLM_SOURCE_DIR}/vllm/v1/worker/gpu_model_runner.py" \
  "${tmpdir}/vllm/v1/worker/gpu_model_runner.py"
cp "${VLLM_SOURCE_DIR}/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py" \
  "${tmpdir}/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"
cp "${VLLM_SOURCE_DIR}/vllm/model_executor/models/glm4_moe_mtp.py" \
  "${tmpdir}/vllm/model_executor/models/glm4_moe_mtp.py"

cat >"${tmpdir}/Dockerfile" <<EOF
FROM ${BASE_IMAGE}

COPY vllm/v1/core/kv_cache_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
COPY vllm/v1/attention/backends/mla/indexer.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py
COPY vllm/v1/spec_decode/step3p5.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/step3p5.py
COPY vllm/v1/spec_decode/llm_base_proposer.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py
COPY vllm/v1/sample/rejection_sampler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py
COPY vllm/v1/worker/gpu_worker.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py
COPY vllm/v1/worker/gpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py
COPY vllm/v1/worker/gpu/spec_decode/rejection_sampler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
COPY vllm/model_executor/models/glm4_moe_mtp.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe_mtp.py

RUN python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/step3p5.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/glm4_moe_mtp.py

LABEL local.kamiwaza.glm52.spark.overlay="kv-mtp-worker-rejection-diagnostics-and-post-load-trim" \
      local.kamiwaza.glm52.spark.base_image="${BASE_IMAGE}"
EOF

echo "Building ${IMAGE} from ${BASE_IMAGE}"
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -t "${IMAGE}" \
  "${tmpdir}"

if [[ "${BUILD_WORKERS}" == "1" ]]; then
  context_archive="${tmpdir}/context.tar"
  tar -C "${tmpdir}" --exclude ./context.tar -cf "${context_archive}" .

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

  for ip in ${WORKER_IPS}; do
    remote_archive="/tmp/glm52-spark-overlay-${USER:-matt}.tar"
    "${scp_base[@]}" "${context_archive}" "${ip}:${remote_archive}"
    "${ssh_base[@]}" "${ip}" "set -euo pipefail
remote_tmp=\$(mktemp -d /tmp/glm52-spark-overlay.XXXXXX)
cleanup() { rm -rf \"\${remote_tmp}\" '${remote_archive}'; }
trap cleanup EXIT
tar -C \"\${remote_tmp}\" -xf '${remote_archive}'
DOCKER_BUILDKIT=1 docker build --progress=plain -t '${IMAGE}' \"\${remote_tmp}\"" &
  done
  wait
fi

echo "Built ${IMAGE}"
