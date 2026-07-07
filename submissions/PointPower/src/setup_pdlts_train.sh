#!/usr/bin/env bash
# PD-LTS fine-tune / training deps (extends setup_pdlts_deps.sh for inference).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PDLTS="${GC2026_ROOT}/code/PD-LTS"
PAIRS="${GC2026_ROOT}/data/processed/train_pairs_official_cgv2.txt"
BASE_CKPT="${PDLTS}/product/ckpt/Denoiseflow-light-FBM.ckpt"

bash "${SCRIPT_DIR}/setup_pdlts_deps.sh"
bash "${SCRIPT_DIR}/download_metric.sh"

if [[ ! -f "$PAIRS" ]]; then
  echo "[pdlts-train] FATAL: missing $PAIRS" >&2
  echo "[pdlts-train] Run: bash ${SUBMISSION_ROOT}/data/generate_pair_lists.sh" >&2
  echo "[pdlts-train] Requires train-split CG+HE PLY under data/raw/UVG-CWI-DQPC/" >&2
  exit 1
fi
n_pairs=$(wc -l < "$PAIRS")
echo "[pdlts-train] train pairs: $n_pairs"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[pdlts-train] FATAL: missing PD-LTS pretrained ckpt: $BASE_CKPT" >&2
  echo "[pdlts-train] Run: bash src/download_pdlts.sh  (clones PD-LTS repo)" >&2
  exit 1
fi

echo "[pdlts-train] pytorch-lightning + tensorboard"
pip install -q 'pytorch-lightning>=2.0,<2.5' tensorboard

echo "[pdlts-train] build EMD CUDA extension"
cd "$PDLTS/metric/emd"
python setup.py install -q

echo "[pdlts-train] import smoke (TrainerModule + EMD + kaolin stub)"
cd "$PDLTS"
python - <<'PY'
import os, sys, types, torch
from pytorch3d.loss import chamfer_distance as p3d_cd

kaolin = types.ModuleType("kaolin")
km = types.ModuleType("kaolin.metrics")
kpc = types.ModuleType("kaolin.metrics.pointcloud")
kpc.chamfer_distance = p3d_cd
km.pointcloud = kpc
kaolin.metrics = km
for name, mod in [("kaolin", kaolin), ("kaolin.metrics", km), ("kaolin.metrics.pointcloud", kpc)]:
    sys.modules[name] = mod

sys.path.insert(0, os.path.join(os.getcwd(), "models"))
sys.path.insert(0, os.getcwd())
_old = torch.load
torch.load = lambda *a, **k: _old(*a, **{**k, "weights_only": False})

from models.model_light.train_deflow_score import TrainerModule, model_specific_args
from metric.loss import EarthMoverDistance1

cfg = model_specific_args().parse_args([])
m = TrainerModule(cfg)
print("[pdlts-train] TrainerModule OK", sum(p.numel() for p in m.network.parameters()))
PY

echo "[pdlts-train] ready."
echo "  Smoke:  bash src/run_pdlts_finetune_uvg.sh smoke"
echo "  Train:  GPUS=4 bash src/run_pdlts_finetune_uvg.sh train"
echo "  Install ckpt for inference: bash src/install_finetuned_ckpt.sh"
