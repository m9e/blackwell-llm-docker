#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a
. ./glm52-mtp1-dcp4-128k.env
set +a
./launch-ray.sh
if [[ "${PATCH_DIAGNOSTICS:-0}" == "1" ]]; then
  ./patch-vllm-diagnostics.sh
fi
./serve.sh
