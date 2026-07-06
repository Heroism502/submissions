#!/usr/bin/env bash
# Official aligned GC-baseline Chamfer (val565 or custom pairs).
#
# Requires: bash src/download_metric.sh  (code/Metric/matrices/)
#
# Usage:
#   OUT_DIR=output/submission_candidate_frame_gate_v2 bash src/run_eval.sh
#   PAIRS_FILE=... OUT_DIR=... bash src/run_eval.sh
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"

OUT="${OUT_DIR:-${GC2026_ROOT}/output/submission_candidate_frame_gate_v2}"
PAIRS="${PAIRS_FILE:-${UVG_VAL_PAIRS_FILE}}"
EVAL_JSON="${EVAL_JSON:-${OUT}/evaluation_gc_baseline_val565.json}"
MAX_FRAMES="${MAX_FRAMES:-0}"

if [[ ! -d "$OUT" ]]; then
  echo "[run_eval] FATAL: OUT_DIR not found: $OUT" >&2
  exit 1
fi
if [[ ! -f "$PAIRS" ]]; then
  echo "[run_eval] FATAL: pairs file not found: $PAIRS" >&2
  echo "[run_eval] Run: bash data/generate_pair_lists.sh" >&2
  exit 1
fi

bash "${SRC_DIR}/download_metric.sh"

extra=()
[[ "$MAX_FRAMES" -gt 0 ]] && extra+=(--max-frames "$MAX_FRAMES")

echo "[run_eval] pairs=$PAIRS out=$OUT gate=$GATE_JSON"
"$PYTHON" "${SRC_DIR}/evaluate_uvg.py" \
  --pairs-file "$PAIRS" \
  --enhanced-root "$OUT" \
  --out-json "$EVAL_JSON" \
  --also-cg \
  "${extra[@]}"

echo "[run_eval] DONE -> $EVAL_JSON"
