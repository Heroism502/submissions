#!/usr/bin/env bash
# Clone UVG-CWI Metric repo (alignment matrices for training / official eval).
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SRC_DIR}/common.sh"
METRIC_ROOT="${METRIC_ROOT:-${GC2026_ROOT}/code/Metric}"
METRIC_REPO="${METRIC_REPO:-https://github.com/UVG-CWI/Metric.git}"
mkdir -p "${GC2026_ROOT}/code"
if [[ ! -d "${METRIC_ROOT}/.git" ]]; then
  git clone "${METRIC_REPO}" "${METRIC_ROOT}"
fi
MAT_DIR="${METRIC_ROOT}/matrices"
if [[ ! -d "$MAT_DIR" ]]; then
  echo "[download_metric] FATAL: missing matrices/ under $METRIC_ROOT" >&2
  exit 1
fi
n=$(find "$MAT_DIR" -maxdepth 1 -name '*.txt' | wc -l)
echo "[download_metric] OK matrices=$n -> $MAT_DIR"
