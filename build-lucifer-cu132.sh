#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Lucifer DS4 Flash/CUTLASS image using the local CUDA 13.2.1 base and cached
# FlashInfer/DeepGEMM build stages. The vLLM branch is built from the rebased
# procr1337 Lucifer patches, not from procr1337's monolithic Dockerfile.

export IMAGE="${IMAGE:-voipmonitor/vllm:lucifer-vllm7c6bbf4-fi3395b41aa8d-dg324aced12c-tk9801a7-cu132-20260609}"
export ALIAS_IMAGE="${ALIAS_IMAGE:-voipmonitor/vllm:lucifer}"

export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:lucifer-cu132-system-base-20260609}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:lucifer-cu132-build-base-20260609}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-1}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"

export MAX_JOBS="${MAX_JOBS:-64}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-refs/pull/3395/head}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-b41aa8dd2fb93c49b1c6134bd1953040f8089d51}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-refs/pull/324/head}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-aced12c2c8882a945c568ace9d4a7e5778aae410}"

export B12X_REPO="${B12X_REPO:-https://github.com/lukealonso/b12x.git}"
export B12X_REF="${B12X_REF:-refs/pull/11/head}"
export B12X_COMMIT="${B12X_COMMIT:-d90d89c8353adabb56cc84bd3924ef811ef8d877}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-lucifer}"
export VLLM_COMMIT="${VLLM_COMMIT:-7c6bbf4c5a482e100af886c5b6eb4303746cc3ba}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+lucifer.cu132.20260609}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"

export TRITON_KERNELS_REPO="${TRITON_KERNELS_REPO:-https://github.com/triton-lang/triton.git}"
export TRITON_KERNELS_REF="${TRITON_KERNELS_REF:-9801a7afbaea43a085db2016eadddd631555ae13}"
export TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT:-9801a7afbaea43a085db2016eadddd631555ae13}"

./build-vllm-b12x-cu132.sh "$@"

if [[ -n "${ALIAS_IMAGE}" ]]; then
  docker tag "${IMAGE}" "${ALIAS_IMAGE}"
fi
