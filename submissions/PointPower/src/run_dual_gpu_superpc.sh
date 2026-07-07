#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
CG_LIST="${CG_LIST:?CG_LIST required}"
GEOMETRY_SECONDARY_DIR="${GEOMETRY_SECONDARY_DIR:-${GC2026_ROOT}/output/superpc_geometry/blend_cg}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NUM_GPUS="${NUM_GPUS:-1}"
[[ "$NUM_GPUS" -lt 1 ]] && NUM_GPUS=1
mkdir -p "$GEOMETRY_SECONDARY_DIR"
SHARD_DIR="${GEOMETRY_SECONDARY_DIR}/.shards_${NUM_GPUS}gpu"
mkdir -p "$SHARD_DIR"
"$PYTHON" "${SRC_DIR}/split_pending_cg_list.py" \
  --cg-list "$CG_LIST" --out-dir "$GEOMETRY_SECONDARY_DIR" --shard-dir "$SHARD_DIR" --num-shards "$NUM_GPUS"
pids=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
  list="${SHARD_DIR}/pending_${i}.txt"
  [[ -s "$list" ]] || continue
  gpu=$((i % NUM_GPUS))
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "${SRC_DIR}/run_superpc_infer.py" \
    --cg-list "$list" --out-dir "$GEOMETRY_SECONDARY_DIR" \
    --ckpt-path "${SUPERPC_CKPT}" --output-mode blend_cg --blend-voxel-mm 3.0 \
    --skip-existing &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[run_dual_gpu_superpc] DONE -> $GEOMETRY_SECONDARY_DIR"
