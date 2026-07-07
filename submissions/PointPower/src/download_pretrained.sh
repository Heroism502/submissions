#!/usr/bin/env bash
# Clone SuperPC repo and ensure kitti360_com.pth exists (bundled or Google Drive download).
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
mkdir -p "${GC2026_ROOT}/code" "${SUBMISSION_ROOT}/models"
if [[ ! -d "${SUPERPC_ROOT}/.git" ]]; then
  git clone https://github.com/sair-lab/SuperPC "${SUPERPC_ROOT}"
fi

CKPT="${SUPERPC_CKPT}"
if [[ -f "$CKPT" ]]; then
  _rel="${CKPT#${SUBMISSION_ROOT}/}"
  [[ "$_rel" == "$CKPT" ]] && _rel="${CKPT#${GC2026_ROOT}/}"
  echo "[download_pretrained] OK ${_rel:-$CKPT}"
  exit 0
fi

DRIVE_URL="https://drive.google.com/drive/folders/1FrQtm8LBVrbdRT4Xs87rIZpJ9nYaTqcG"
OUT_DIR="${SUBMISSION_ROOT}/models"
echo "[download_pretrained] Missing $CKPT — downloading kitti360_com.pth from SuperPC Model Zoo..."

if timeout 8 curl -fsI --connect-timeout 5 https://drive.google.com >/dev/null 2>&1; then
  pip install -q gdown
  for attempt in 1 2 3; do
    timeout_sec=$((120 * attempt))
    echo "[download_pretrained] gdown attempt $attempt (timeout ${timeout_sec}s)"
    if timeout "$timeout_sec" gdown --folder "$DRIVE_URL" -O "$OUT_DIR" --remaining-ok 2>&1; then
      break
    fi
    sleep 5
  done
fi

if [[ -f "${OUT_DIR}/kitti360_com.pth" ]]; then
  echo "[download_pretrained] OK models/kitti360_com.pth"
  exit 0
fi

echo "[download_pretrained] FATAL: kitti360_com.pth not found after download." >&2
echo "[download_pretrained] Manual: download from $DRIVE_URL -> $OUT_DIR/kitti360_com.pth" >&2
exit 1
