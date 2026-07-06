#!/usr/bin/env bash
# Enhancement Only: CG -> ft PD-LTS -> SuperPC blend_cg -> frame_gate v2 hybrid refine.
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"

export OUT_DIR="${OUT_DIR:-${GC2026_ROOT}/output/submission_candidate_frame_gate_v2}"
export GEOMETRY_DIR="${GEOMETRY_DIR:-${GC2026_ROOT}/output/pdlts_finetune_geometry/light}"
export GEOMETRY_SECONDARY_DIR="${GEOMETRY_SECONDARY_DIR:-${GC2026_ROOT}/output/superpc_geometry/blend_cg}"
export UVG_CG_VERSION="${UVG_CG_VERSION:-v2}"
export SKIP_PDLTS="${SKIP_PDLTS:-0}"
export SKIP_SUPERPC="${SKIP_SUPERPC:-0}"

if [[ -z "${CG_LIST:-}" ]]; then
  echo "[run.sh] FATAL: CG_LIST is required. Export the path to your CG PLY list." >&2
  echo "  export CG_LIST=\$GC2026_ROOT/data/processed/all_cg_only_cgv2.txt" >&2
  exit 1
fi
export CG_LIST

"${SRC_DIR}/download_pdlts.sh"
"${SRC_DIR}/download_pretrained.sh"

if [[ "$SKIP_PDLTS" != "1" ]]; then
  echo "[run.sh] Stage1 ft PD-LTS -> $GEOMETRY_DIR"
  bash "${SRC_DIR}/run_dual_gpu_pdlts.sh"
fi
if [[ "$SKIP_SUPERPC" != "1" ]]; then
  echo "[run.sh] Stage2 SuperPC blend_cg -> $GEOMETRY_SECONDARY_DIR"
  bash "${SRC_DIR}/run_dual_gpu_superpc.sh"
fi

echo "[run.sh] Stage3 frame_gate v2 hybrid -> $OUT_DIR"
if [[ "${NUM_SHARDS:-1}" -gt 1 ]]; then
  bash "${SRC_DIR}/run_enh_refine_sharded.sh"
else
  "$PYTHON" "${SRC_DIR}/run_enh_refine_infer.py" \
    --cg-list "$CG_LIST" --out-dir "$OUT_DIR" \
    --refine-config "$GATE_JSON" \
    --geometry-dir "$GEOMETRY_DIR" \
    --geometry-secondary-dir "$GEOMETRY_SECONDARY_DIR" \
    --use-geometry-cache --require-geometry-cache
fi

echo "[run.sh] DONE -> $OUT_DIR"
