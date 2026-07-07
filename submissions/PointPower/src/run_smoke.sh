#!/usr/bin/env bash
# Quick smoke test on 2 frames (fresh pipeline; clears prior smoke outputs).
# Prerequisite: pair lists generated — see data/DATA_LAYOUT.md
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
SMOKE_FRAMES="${SMOKE_FRAMES:-2}"
OUT_DIR="${OUT_DIR:-${GC2026_ROOT}/output/submission_smoke_frame_gate_v2}"
VAL_CG="${GC2026_ROOT}/data/processed/val_cg_only_official_cgv2.txt"
CLEAN_SMOKE="${CLEAN_SMOKE:-1}"

if [[ ! -f "$VAL_CG" ]]; then
  echo "[run_smoke] Pair lists not found: data/processed/val_cg_only_official_cgv2.txt" >&2
  echo "[run_smoke] Configure data/splits/val/<Seq>/ markers then: bash data/generate_pair_lists.sh" >&2
  exit 1
fi
if [[ ! -f "${SUBMISSION_ROOT}/models/DenoiseFlow-light-UVG-finetune.ckpt" ]]; then
  echo "[run_smoke] FATAL: bundled ckpt missing: models/DenoiseFlow-light-UVG-finetune.ckpt" >&2
  exit 1
fi
if [[ ! -f "${SUBMISSION_ROOT}/models/kitti360_com.pth" ]]; then
  echo "[run_smoke] FATAL: bundled ckpt missing: models/kitti360_com.pth" >&2
  exit 1
fi

if [[ "$CLEAN_SMOKE" == "1" ]]; then
  echo "[run_smoke] cleaning $OUT_DIR"
  rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"
head -n "$SMOKE_FRAMES" "$VAL_CG" > "$OUT_DIR/smoke_cg_list.txt"
export OUT_DIR CG_LIST="$OUT_DIR/smoke_cg_list.txt"
export GEOMETRY_DIR="${GEOMETRY_DIR:-$OUT_DIR/pdlts_geometry}"
export GEOMETRY_SECONDARY_DIR="${GEOMETRY_SECONDARY_DIR:-$OUT_DIR/superpc_geometry}"
# GATE_JSON from common.sh (runtime_gate.json if generate_pair_lists ran)
bash "${SRC_DIR}/run.sh"
n=$(find "$OUT_DIR" -maxdepth 2 -name '*_ENH_*.ply' 2>/dev/null | wc -l)
echo "[run_smoke] ENH PLY count=$n (expected $SMOKE_FRAMES)"
if [[ "$n" -lt "$SMOKE_FRAMES" ]]; then
  echo "[run_smoke] FAIL: expected at least $SMOKE_FRAMES ENH files under output/submission_smoke_frame_gate_v2/" >&2
  exit 1
fi

VAL_PAIRS="${GC2026_ROOT}/data/processed/val_pairs_official_cgv2.txt"
RUN_SMOKE_EVAL="${RUN_SMOKE_EVAL:-1}"
if [[ "$RUN_SMOKE_EVAL" == "1" && -f "$VAL_PAIRS" ]]; then
  head -n "$SMOKE_FRAMES" "$VAL_PAIRS" > "$OUT_DIR/smoke_val_pairs.txt"
  echo "[run_smoke] official Chamfer on $SMOKE_FRAMES val frames"
  PAIRS_FILE="$OUT_DIR/smoke_val_pairs.txt" \
    EVAL_JSON="$OUT_DIR/evaluation_smoke_gc_baseline.json" \
    MAX_FRAMES=0 \
    bash "${SRC_DIR}/run_eval.sh"
elif [[ "$RUN_SMOKE_EVAL" == "1" ]]; then
  echo "[run_smoke] WARN: skip eval — missing $VAL_PAIRS (need HE + generate_pair_lists)" >&2
fi

echo "[run_smoke] DONE -> output/submission_smoke_frame_gate_v2/"
