#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
mkdir -p "${GC2026_ROOT}/code"
if [[ ! -d "${PDLTS_ROOT}/.git" ]]; then
  git clone https://github.com/yanbiao1/PD-LTS.git "${PDLTS_ROOT}"
fi
BASE_CKPT="${PDLTS_ROOT}/product/ckpt/Denoiseflow-light-FBM.ckpt"
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[download_pdlts] WARN: official pretrained ckpt not found: $BASE_CKPT" >&2
  echo "[download_pdlts]        Required for fine-tune; inference uses bundled UVG ckpt." >&2
fi
if [[ ! -f "${PDLTS_FINETUNE_CKPT}" ]]; then
  echo "Missing finetune ckpt: ${PDLTS_FINETUNE_CKPT}" >&2
  exit 1
fi
if [[ "$PDLTS_FINETUNE_CKPT" == "${SUBMISSION_ROOT}/models/"* ]]; then
  echo "[download_pdlts] PD-LTS ckpt: models/DenoiseFlow-light-UVG-finetune.ckpt (bundled)"
else
  rel="${PDLTS_FINETUNE_CKPT#${GC2026_ROOT}/}"
  echo "[download_pdlts] PD-LTS ckpt: ${rel:-$PDLTS_FINETUNE_CKPT} (self-trained, auto-detected)"
fi
echo "[download_pdlts] finetune ckpt OK"
