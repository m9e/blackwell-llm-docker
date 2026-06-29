#!/usr/bin/env bash
set -euo pipefail

# Build the local model directory used by this recipe from public Hugging Face
# repos. This avoids requiring a separately uploaded monolithic hybrid repo.

BASE_REPO="${BASE_REPO:-Mapika/GLM-5.2-NVFP4}"
MTP_REPO="${MTP_REPO:-sant1an/GLM-5.2-NVFP4-MTP}"
MODEL_ROOT="${MODEL_ROOT:-/var/tmp/models}"

BASE_DIR="${BASE_DIR:-${MODEL_ROOT}/Mapika/GLM-5.2-NVFP4}"
MTP_DIR="${MTP_DIR:-${MODEL_ROOT}/sant1an/GLM-5.2-NVFP4-MTP}"
HYBRID_DIR="${HYBRID_DIR:-${MODEL_ROOT}/Mapika/GLM-5.2-NVFP4-MTP-hybrid}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required. Install huggingface_hub first." >&2
  exit 2
fi

mkdir -p "${MODEL_ROOT}"

echo "Downloading base checkpoint: ${BASE_REPO}"
huggingface-cli download "${BASE_REPO}" --local-dir "${BASE_DIR}"

echo "Downloading MTP overlay: ${MTP_REPO}"
huggingface-cli download "${MTP_REPO}" --local-dir "${MTP_DIR}"

echo "Creating hybrid directory: ${HYBRID_DIR}"
mkdir -p "${HYBRID_DIR}"

if cp -al "${BASE_DIR}/." "${HYBRID_DIR}/" 2>/dev/null; then
  echo "Base checkpoint hardlinked into hybrid directory."
else
  echo "Hardlink copy failed; falling back to full copy."
  cp -a "${BASE_DIR}/." "${HYBRID_DIR}/"
fi

cp -a "${MTP_DIR}/." "${HYBRID_DIR}/"

python3 - "${HYBRID_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index_path = root / "model.safetensors.index.json"
if not index_path.exists():
    raise SystemExit(f"missing {index_path}")

index = json.loads(index_path.read_text())
weight_map = index.get("weight_map", {})
layer78 = {
    key: value
    for key, value in weight_map.items()
    if key.startswith("model.layers.78.")
}
if not layer78:
    raise SystemExit("hybrid index has no model.layers.78.* MTP weights")

missing = sorted({value for value in layer78.values() if not (root / value).exists()})
if missing:
    raise SystemExit("hybrid index references missing MTP shard(s): " + ", ".join(missing))

print(f"OK: {len(layer78)} layer-78 MTP tensors present in {root}")
PY

echo "Hybrid checkpoint ready: ${HYBRID_DIR}"
