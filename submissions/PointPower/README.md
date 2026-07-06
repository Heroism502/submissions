# UVG-CWI-DQPC Grand Challenge at ACM Multimedia 2026: Team PointPower (Enhancement Only)

## Team Name
PointPower

## Team Members
- Tianhua Qi (qitianhua@seu.edu.cn) — Southeast University
- Jiateng Liu (jiateng_liu@seu.edu.cn) — Southeast University
- Mingxin Tang (antonytang@hnu.edu.cn) — Hunan University
- Tianyi Zhang (t.zhang@seu.edu.cn) — Southeast University
- Yuan Zong (xhzongyuan@seu.edu.cn) — Southeast University
- Hengcan Shi (shihengcan@hnu.edu.cn) — Hunan University
- Yaonan Wang (yaonan@hnu.edu.cn) — Hunan University
- Wenming Zheng (wenming_zheng@seu.edu.cn) — Southeast University

## Algorithm Name
PointPower

## Algorithm Description
We propose **PointPower**, a progressive multi-stage pipeline tailored for dynamic point cloud enhancement. The first stage introduces an invertible latent-space denoising module that maps corrupted point clouds into a well-structured latent space for robust, topology-aware noise filtering. The second stage performs spatial anchoring and primary density refinement to regularize point coordinates and density in Euclidean space, yielding a reliable primary representation. The third stage executes a decision-driven gated refinement, utilizing a frame-level gating mechanism to dynamically invoke a pretrained diffusion model for high-fidelity structural completion only when significant geometric deficits are detected. Extensive experiments on the UVG-CWI-DQPC Challenge 2026 dataset demonstrate superior geometric fidelity on real-world 4D volumetric data.

## Processing Track
**Enhancement Only** — Improve the already-extracted consumer-grade `.ply` point clouds (official CG v2, 15 fps).

## Hardware / Environment
- GPU: 4 × NVIDIA RTX 5090 (CUDA 12.8); smoke test works on 1 GPU
- OS: Ubuntu 22.04 LTS
- Python: 3.9 (conda env `superpc`)

## Runtime
~9 hours for full inference on 2155 frames (PD-LTS ~5 h + SuperPC ~3 h + refine ~1 h on 2 GPUs each for Stages 1–2).

## How to Run

Data layout and train/val split: [`data/DATA_LAYOUT.md`](data/DATA_LAYOUT.md), [`data/splits/README.md`](data/splits/README.md).  
Full setup: [`SETUP.md`](SETUP.md).

```bash
# Extract to any location; GC2026_ROOT = workspace containing data/raw/ (not the submission folder itself)
tar xzf GC2026_Team_EnhancementOnly.tar.gz && cd GC2026_Team_EnhancementOnly
export GC2026_ROOT=../workspace

# Place train/val markers (see data/splits/README.md), then:
bash data/generate_pair_lists.sh

conda create -n superpc python=3.9 -y && conda activate superpc
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
bash src/download_pdlts.sh && bash src/download_pretrained.sh && bash src/setup_pdlts_deps.sh

bash src/run_smoke.sh          # 2-frame smoke + official aligned Chamfer (if val HE pairs exist)
export CG_LIST=$GC2026_ROOT/data/processed/all_cg_only_cgv2.txt
bash src/run.sh                # full 2155-frame inference
bash src/post_submission_candidate.sh   # manifest + val565 evaluation
```

**PD-LTS weights:** if `output/pdlts_finetune_uvg/run_*/DenoiseFlow-light-UVG-finetune.ckpt` exists, it is used automatically; otherwise the bundled `models/DenoiseFlow-light-UVG-finetune.ckpt` is used. Pretrained checkpoints under `models/` are required at inference.

`run_smoke.sh` runs `run_eval.sh` on 2 val frames by default (`RUN_SMOKE_EVAL=0` to skip). The reported **14.870 mm** requires bundled ckpt + full val565 + `post_submission_candidate.sh` (Metric alignment + official Chamfer via `evaluate_uvg.py`).

## PD-LTS Fine-tuning (Optional)

When train CG+HE pairs are available:

```bash
bash src/setup_pdlts_train.sh
bash src/run_pdlts_finetune_uvg.sh smoke
GPUS=4 bash src/run_pdlts_finetune_uvg.sh train
bash src/install_finetuned_ckpt.sh
```

Train/val split is written by `bash data/generate_pair_lists.sh` from `data/raw/.../train|val/` or `data/splits/val/<Seq>/` markers into `data/splits/split.json`. Fine-tuning warm-starts from public PD-LTS FBM pretrained weights.

## Local Validation (val565 — TrumanShow, VictoryHeart, VirtualLife; 564 frames)

| Method | Chamfer Distance (mm) ↓ |
|--------|--------------------------:|
| ft PD-LTS + density (no SuperPC) | 14.883 |
| **PointPower (ours)** | **14.870** |
| Official CG baseline | 17.552 |

Evaluated with `bash src/post_submission_candidate.sh` → `evaluate_uvg.py` (Metric alignment matrices + official Chamfer). Preset: `config/gate_decision.json` / `data/processed/runtime_gate.json`.
