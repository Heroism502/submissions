#!/usr/bin/env bash
# After full inference: manifest + official val565 Chamfer evaluation.
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
OUT="${OUT_DIR:-${GC2026_ROOT}/output/submission_candidate_frame_gate_v2}"
export OUT_DIR="$OUT"
VAL_PAIRS="${UVG_VAL_PAIRS_FILE}"
META="${GC2026_ROOT}/data/processed/split_meta_official_cgv2.json"
DATA_SPLIT="organizer"
if [[ -f "$META" ]]; then
  DATA_SPLIT=$("$PYTHON" -c "import json; m=json.load(open('${META}')); v=m.get('val_sequences',[]); print('val='+','.join(v) if v else 'organizer')")
fi
n=$(find "$OUT" -name '*_ENH_*.ply' 2>/dev/null | wc -l)
echo "[post] enh_ply_count=$n gate=$GATE_JSON"
"$PYTHON" "${SRC_DIR}/write_runtime_summary.py" --out-dir "$OUT" --team "GC2026 Team" || true
"$PYTHON" "${SRC_DIR}/make_submission.py" \
  --enhanced-dir "$OUT" --team "GC2026 Team" \
  --processing-track "Enhancement Only" \
  --title "UVG-CWI-DQPC GC2026 Enhancement Only frame_gate v2 hybrid" \
  --post-processing "$GATE_JSON" --cg-version "${UVG_CG_VERSION:-v2}" \
  --cg-source "official" \
  --data-split "$DATA_SPLIT" \
  --pipeline-notes "CGv2 -> UVG-finetuned PD-LTS light -> SuperPC blend_cg secondary -> frame_gate v2 (primary anchor, CG-hole mask, max10pct fill)"
if [[ ! -f "$VAL_PAIRS" ]]; then
  echo "[post] FATAL: val pairs not found: $VAL_PAIRS" >&2
  echo "[post] Run: bash data/generate_pair_lists.sh (requires CG+HE for val sequences)" >&2
  exit 1
fi
PAIRS_FILE="$VAL_PAIRS" \
  EVAL_JSON="${OUT}/evaluation_gc_baseline_val565.json" \
  bash "${SRC_DIR}/run_eval.sh"
echo "[post] DONE -> $OUT"
