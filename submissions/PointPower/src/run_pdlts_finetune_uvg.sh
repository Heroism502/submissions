#!/usr/bin/env bash
# PD-LTS UVG fine-tune — adapts to train_pairs file size (any number of sequences/frames).
#
# Prerequisites:
#   bash data/generate_pair_lists.sh   # or set PAIRS_FILE to custom CG/HE list
#   bash src/download_pdlts.sh
#   bash src/setup_pdlts_train.sh
#
#   bash src/run_pdlts_finetune_uvg.sh smoke
#   GPUS=4 bash src/run_pdlts_finetune_uvg.sh train
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
MODE="${1:-smoke}"
GPUS="${GPUS:-4}"
OUT="${OUT_DIR:-${GC2026_ROOT}/output/pdlts_finetune_uvg}"
PAIRS="${PAIRS_FILE:-${GC2026_ROOT}/data/processed/train_pairs_official_cgv2.txt}"
LOG="$OUT/logs"
mkdir -p "$LOG"

if [[ ! -f "$PAIRS" ]]; then
  echo "[finetune] FATAL: missing pairs file (relative: data/processed/train_pairs_official_cgv2.txt)" >&2
  echo "[finetune] Run: bash data/generate_pair_lists.sh" >&2
  exit 1
fi
n=$(wc -l < "$PAIRS")
if [[ "$n" -lt 1 ]]; then
  echo "[finetune] FATAL: pairs file is empty: $PAIRS" >&2
  exit 1
fi
echo "[finetune] train pairs: $n (from $(basename "$PAIRS"))"

# ~5 patches per train frame; override with PATCHES_PER_EPOCH
_default_patches=$(( n * 5 ))
[[ "$_default_patches" -lt 256 ]] && _default_patches=256

smoke() {
  local smoke_frames=16
  [[ "$n" -lt "$smoke_frames" ]] && smoke_frames="$n"
  echo "[finetune] smoke: ${smoke_frames} frames, 8 batches, 1 epoch, 1 GPU"
  python "${SCRIPT_DIR}/run_pdlts_finetune_uvg.py" \
    --pairs-file "$PAIRS" \
    --out-dir "$OUT/smoke" \
    --max-train-frames "$smoke_frames" \
    --patches-per-epoch 64 \
    --batch-size 2 \
    --max-epochs 1 \
    --gpus 1 \
    --limit-train-batches 8 \
    --num-workers 2 \
    2>&1 | tee "$LOG/smoke.log"
}

train() {
  MAX_EPOCHS="${MAX_EPOCHS:-20}"
  PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-$_default_patches}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  LR="${LEARNING_RATE:-5e-4}"
  STOP_LOSS="${STOP_LOSS:-0}"
  STOP_PATIENCE="${STOP_PATIENCE:-0}"
  echo "[finetune] full train: pairs=$n GPUS=$GPUS epochs=$MAX_EPOCHS patches=$PATCHES_PER_EPOCH batch=$BATCH_SIZE -> $OUT"
  extra_args=()
  [[ "$STOP_LOSS" != "0" ]] && extra_args+=(--early-stop-loss "$STOP_LOSS")
  [[ "$STOP_PATIENCE" != "0" ]] && extra_args+=(--early-stop-patience "$STOP_PATIENCE")
  python "${SCRIPT_DIR}/run_pdlts_finetune_uvg.py" \
    --pairs-file "$PAIRS" \
    --out-dir "$OUT/run_$(date +%Y%m%d_%H%M%S)" \
    --patches-per-epoch "$PATCHES_PER_EPOCH" \
    --batch-size "$BATCH_SIZE" \
    --max-epochs "$MAX_EPOCHS" \
    --learning-rate "$LR" \
    --gpus "$GPUS" \
    --num-workers 8 \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG/train_${GPUS}gpu.log"
}

case "$MODE" in
  smoke) smoke ;;
  train) train ;;
  *)
    echo "Usage: $0 {smoke|train}" >&2
    exit 1
    ;;
esac
