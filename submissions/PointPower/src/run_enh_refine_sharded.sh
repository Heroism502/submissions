#!/usr/bin/env bash
# Stage3 refine: split CG list across CPU shards (geometry-cache path is CPU-bound).
#
# Usage (same env as run.sh):
#   export CG_LIST=... OUT_DIR=... GEOMETRY_DIR=... GEOMETRY_SECONDARY_DIR=...
#   NUM_SHARDS=16 bash src/run_enh_refine_sharded.sh
#
# frame_gate v2 (temporal_window=0): shards are independent per frame.
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
# shellcheck source=cpu_parallel_defaults.sh
source "${SRC_DIR}/cpu_parallel_defaults.sh"

CG_LIST="${CG_LIST:?CG_LIST required}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
GEOMETRY_DIR="${GEOMETRY_DIR:-${GC2026_ROOT}/output/pdlts_finetune_geometry/light}"
GEOMETRY_SECONDARY_DIR="${GEOMETRY_SECONDARY_DIR:-${GC2026_ROOT}/output/superpc_geometry/blend_cg}"
LOGDIR="${LOGDIR:-$(dirname "$OUT_DIR")/logs}"
SHARD_DIR="${SHARD_DIR:-$LOGDIR/shards_$(basename "$OUT_DIR")}"

mkdir -p "$OUT_DIR" "$LOGDIR" "$SHARD_DIR"
echo "[run_enh_refine_sharded] NUM_SHARDS=$NUM_SHARDS OMP=$OMP_NUM_THREADS OUT=$OUT_DIR"

"$PYTHON" - <<PY
import os
cg_list = "$CG_LIST"
shard_dir = "$SHARD_DIR"
n = int("$NUM_SHARDS")
paths = []
with open(cg_list, encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        paths.append(ln.split("\t")[0])
buckets = [[] for _ in range(n)]
for i, p in enumerate(paths):
    buckets[i % n].append(p)
os.makedirs(shard_dir, exist_ok=True)
for sid, bucket in enumerate(buckets):
    with open(os.path.join(shard_dir, f"cg_shard_{sid}.txt"), "w", encoding="utf-8") as out:
        out.write("\n".join(bucket) + ("\n" if bucket else ""))
    print(f"[shard] shard_{sid}: {len(bucket)} frames")
PY

INFER_ARGS=(
  --refine-config "$GATE_JSON"
  --geometry-dir "$GEOMETRY_DIR"
  --geometry-secondary-dir "$GEOMETRY_SECONDARY_DIR"
  --use-geometry-cache
  --require-geometry-cache
  --out-dir "$OUT_DIR"
  --no-save-config
)

PIDS=()
for ((sid = 0; sid < NUM_SHARDS; sid++)); do
  shard_list="$SHARD_DIR/cg_shard_${sid}.txt"
  [[ -s "$shard_list" ]] || continue
  log="$LOGDIR/$(basename "$OUT_DIR")_shard${sid}.log"
  echo "[run_enh_refine_sharded] shard${sid} -> $log"
  (
    "$PYTHON" "${SRC_DIR}/run_enh_refine_infer.py" \
      --cg-list "$shard_list" \
      "${INFER_ARGS[@]}" \
      > "$log" 2>&1
  ) &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done
if [[ "$FAIL" -ne 0 ]]; then
  echo "[run_enh_refine_sharded] FAILED — see $LOGDIR/*_shard*.log" >&2
  exit 1
fi

# Merge infer_meta.json (skip re-refine existing PLY)
"$PYTHON" "${SRC_DIR}/run_enh_refine_infer.py" \
  --cg-list "$CG_LIST" \
  --out-dir "$OUT_DIR" \
  --refine-config "$GATE_JSON" \
  --geometry-dir "$GEOMETRY_DIR" \
  --geometry-secondary-dir "$GEOMETRY_SECONDARY_DIR" \
  --use-geometry-cache \
  --require-geometry-cache \
  --skip-existing \
  2>&1 | tail -3

echo "[run_enh_refine_sharded] DONE -> $OUT_DIR"
