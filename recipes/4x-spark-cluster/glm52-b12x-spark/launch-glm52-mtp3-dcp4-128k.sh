#!/usr/bin/env bash
set -euo pipefail

# Launch the validated GLM-5.2 MTP3/DCP4/128K Spark configuration.
# This restarts the recipe's Ray containers, patches in local diagnostics if
# requested, and starts vLLM serving from the known-good env file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/glm52-mtp3-dcp4-128k.env}"
PATCH_DIAGNOSTICS="${PATCH_DIAGNOSTICS:-0}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 2
fi

set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

"${SCRIPT_DIR}/launch-ray.sh"

if [[ "${PATCH_DIAGNOSTICS}" == "1" ]]; then
  "${SCRIPT_DIR}/patch-vllm-diagnostics.sh"
fi

"${SCRIPT_DIR}/serve.sh"
