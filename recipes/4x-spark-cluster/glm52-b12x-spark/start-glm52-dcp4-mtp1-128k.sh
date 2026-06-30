#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/glm52-dcp4-mtp1-128k.env}"
PATCH_DIAGNOSTICS="${PATCH_DIAGNOSTICS:-1}"

exec "${SCRIPT_DIR}/launch-glm52-mtp3-dcp4-128k.sh"
