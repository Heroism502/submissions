#!/usr/bin/env bash
# Copy a fine-tuned PD-LTS checkpoint into submission models/ for inference.
#
# Usage:
#   bash src/install_finetuned_ckpt.sh /path/to/DenoiseFlow-light-UVG-finetune.ckpt
#   bash src/install_finetuned_ckpt.sh   # auto-pick latest run under output/pdlts_finetune_uvg/
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
DEST="${SUBMISSION_ROOT}/models/DenoiseFlow-light-UVG-finetune.ckpt"
SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  SRC=$(ls -t "${GC2026_ROOT}/output/pdlts_finetune_uvg"/run_*/DenoiseFlow-light-UVG-finetune.ckpt 2>/dev/null | head -1 || true)
fi
if [[ -z "$SRC" || ! -f "$SRC" ]]; then
  echo "[install_finetuned_ckpt] FATAL: checkpoint not found. Train first or pass path." >&2
  exit 1
fi
mkdir -p "${SUBMISSION_ROOT}/models"
cp -f "$SRC" "$DEST"
echo "[install_finetuned_ckpt] installed -> $DEST"
