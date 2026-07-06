#!/usr/bin/env bash
# Generate CG/HE pair lists — auto-read folder names, write split.json + runtime_gate.json
set -euo pipefail
GC2026_ROOT="${GC2026_ROOT:-./workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export GC2026_ROOT SUBMISSION_ROOT
python3 "${SCRIPT_DIR}/generate_pair_lists.py"
