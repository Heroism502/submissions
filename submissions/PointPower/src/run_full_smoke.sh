#!/usr/bin/env bash
# Ordered smoke: auto split from folders → lists → runtime gate → 2-frame infer.
set -euo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GC2026_ROOT="${GC2026_ROOT:?set GC2026_ROOT}"
export SUBMISSION_ROOT="$SUB"
cd "$SUB"

echo "=== [1/4] generate_pair_lists (folder names → split.json + runtime_gate.json) ==="
bash data/generate_pair_lists.sh

echo "=== [2/4] bundled weights ==="
test -f models/DenoiseFlow-light-UVG-finetune.ckpt
test -f models/kitti360_com.pth
test -f "$GC2026_ROOT/data/processed/runtime_gate.json"
echo "OK"

echo "=== [3/4] upstream clone check ==="
bash src/download_pdlts.sh
bash src/download_pretrained.sh

echo "=== [4/4] run_smoke (2-frame infer + official Chamfer via run_eval.sh) ==="
export CLEAN_SMOKE=1
export PDLTS_FINETUNE_CKPT="$SUB/models/DenoiseFlow-light-UVG-finetune.ckpt"
bash src/run_smoke.sh

echo "=== ALL SMOKE CHECKS PASSED (infer + metric eval) ==="
