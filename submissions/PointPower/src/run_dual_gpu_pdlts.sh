#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
CG_LIST="${CG_LIST:?CG_LIST required}"
GEOMETRY_DIR="${GEOMETRY_DIR:-${GC2026_ROOT}/output/pdlts_finetune_geometry/light}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-1}"
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
mkdir -p "$GEOMETRY_DIR"
if [[ "$PDLTS_FINETUNE_CKPT" == "${SUBMISSION_ROOT}/"* ]]; then
  _ckpt_rel="${PDLTS_FINETUNE_CKPT#${SUBMISSION_ROOT}/}"
else
  _ckpt_rel="${PDLTS_FINETUNE_CKPT#${GC2026_ROOT}/}"
  [[ "$_ckpt_rel" == "$PDLTS_FINETUNE_CKPT" ]] && _ckpt_rel="$(basename "$PDLTS_FINETUNE_CKPT")"
fi
echo "[run_dual_gpu_pdlts] using ckpt: $_ckpt_rel"
SHARD_DIR="${GEOMETRY_DIR}/.shards_${NUM_GPUS}gpu"
mkdir -p "$SHARD_DIR"
"$PYTHON" "${SRC_DIR}/split_pending_cg_list.py" \
  --cg-list "$CG_LIST" --out-dir "$GEOMETRY_DIR" --shard-dir "$SHARD_DIR" --num-shards "$NUM_GPUS"
pids=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
  list="${SHARD_DIR}/pending_${i}.txt"
  [[ -s "$list" ]] || continue
  gpu=$((i % NUM_GPUS))
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "${SRC_DIR}/run_pdlts_infer.py" \
    --cg-list "$list" --out-dir "$GEOMETRY_DIR" --model light \
    --ckpt "${PDLTS_FINETUNE_CKPT}" --skip-existing &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[run_dual_gpu_pdlts] DONE -> $GEOMETRY_DIR"
