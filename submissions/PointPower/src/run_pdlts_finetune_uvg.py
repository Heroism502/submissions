#!/usr/bin/env python3
"""Fine-tune PD-LTS DenoiseFlow-light on UVG train CG/HE pairs (train split from data/splits/)."""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch3d.loss import chamfer_distance as p3d_cd
from pytorch_lightning.callbacks import EarlyStopping

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from gc2026_paths import resolve_gc2026_root  # noqa: E402

GC2026_ROOT = resolve_gc2026_root(SCRIPT_DIR)
PDLTS_ROOT = os.path.join(GC2026_ROOT, "code", "PD-LTS")
DEFAULT_PAIRS = os.path.join(GC2026_ROOT, "data/processed/train_pairs_official_cgv2.txt")
DEFAULT_CKPT = os.path.join(PDLTS_ROOT, "product/ckpt/Denoiseflow-light-FBM.ckpt")

sys.path.insert(0, SCRIPT_DIR)
from pdlts_uvg_train_dataset import build_train_loader, summarize_train_pairs  # noqa: E402


def setup_pdlts_train_env() -> None:
    kaolin = types.ModuleType("kaolin")
    km = types.ModuleType("kaolin.metrics")
    kpc = types.ModuleType("kaolin.metrics.pointcloud")
    kpc.chamfer_distance = p3d_cd
    km.pointcloud = kpc
    kaolin.metrics = km
    for name, mod in [
        ("kaolin", kaolin),
        ("kaolin.metrics", km),
        ("kaolin.metrics.pointcloud", kpc),
    ]:
        sys.modules[name] = mod

    if PDLTS_ROOT not in sys.path:
        sys.path.insert(0, PDLTS_ROOT)
    sys.path.insert(0, os.path.join(PDLTS_ROOT, "models"))
    os.chdir(PDLTS_ROOT)

    _orig = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load_compat  # type: ignore[method-assign]


class UvgFinetuneDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        return build_train_loader(
            self.args.pairs_file,
            max_frames=self.args.max_train_frames,
            max_points=self.args.max_points,
            patch_size=self.args.patch_size,
            patches_per_epoch=self.args.patches_per_epoch,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune PD-LTS on UVG train CG/HE pairs")
    p.add_argument("--pairs-file", default=DEFAULT_PAIRS)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--out-dir", default=os.path.join(GC2026_ROOT, "output/pdlts_finetune_uvg"))
    p.add_argument("--max-train-frames", type=int, default=0, help="0 = all train pairs")
    p.add_argument("--max-points", type=int, default=30_000)
    p.add_argument("--patch-size", type=int, default=1024)
    p.add_argument("--patches-per-epoch", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--limit-train-batches", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--early-stop-loss",
        type=float,
        default=0.0,
        help="Stop when epoch-average train loss <= this value (0 = disabled)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Also stop if train loss does not improve for N epochs (0 = disabled)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    info = summarize_train_pairs(args.pairs_file, max_frames=args.max_train_frames)
    if info["num_pairs"] == 0:
        raise SystemExit(f"No training pairs in {args.pairs_file}")
    if info["missing_matrices"]:
        raise SystemExit(
            "Missing Metric alignment matrices for: "
            + ", ".join(info["missing_matrices"])
            + f"\n  Expected under: code/Metric/matrices/\n  Run: bash src/download_metric.sh"
        )
    print(
        f"[finetune] pairs={info['num_pairs']} sequences={len(info['sequences'])}: "
        + ", ".join(info["sequences"])
    )
    setup_pdlts_train_env()

    from models.model_light.train_deflow_score import (  # noqa: WPS433
        TrainerModule,
        model_specific_args,
    )

    class UvgFinetuneModule(TrainerModule):
        automatic_optimization = False

        def on_train_epoch_end(self):
            if self.training_step_outputs:
                epoch_avg = float(torch.mean(torch.tensor(self.training_step_outputs)))
                self.log(
                    "train_loss_epoch",
                    epoch_avg,
                    prog_bar=True,
                    sync_dist=True,
                )
            super().on_train_epoch_end()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = model_specific_args().parse_args([])
    cfg.learning_rate = args.learning_rate

    module = UvgFinetuneModule(cfg)
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"Missing checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    module.network.load_state_dict(state)
    print(f"[finetune] loaded pretrained {args.ckpt}")

    dm = UvgFinetuneDataModule(args)

    devices = args.gpus if args.gpus > 0 else 1
    trainer_kw = dict(
        default_root_dir=str(out_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        max_epochs=args.max_epochs,
        precision=32,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        logger=True,
        fast_dev_run=args.fast_dev_run,
    )
    if args.limit_train_batches > 0:
        trainer_kw["limit_train_batches"] = args.limit_train_batches

    callbacks: list[EarlyStopping] = []
    if args.early_stop_loss > 0 or args.early_stop_patience > 0:
        es_kw = dict(
            monitor="train_loss_epoch",
            mode="min",
            check_on_train_epoch_end=True,
            verbose=True,
        )
        if args.early_stop_loss > 0:
            es_kw["stopping_threshold"] = args.early_stop_loss
        if args.early_stop_patience > 0:
            es_kw["patience"] = args.early_stop_patience
        else:
            es_kw["patience"] = args.max_epochs
        callbacks.append(EarlyStopping(**es_kw))
        print(
            f"[finetune] early stop: loss<={args.early_stop_loss or '—'} "
            f"patience={args.early_stop_patience or 'off'}"
        )
    if callbacks:
        trainer_kw["callbacks"] = callbacks

    pl.seed_everything(args.seed, workers=True)
    trainer = pl.Trainer(**trainer_kw)
    trainer.fit(module, datamodule=dm)

    save_path = out_dir / "DenoiseFlow-light-UVG-finetune.ckpt"
    torch.save(module.network.state_dict(), save_path)
    print(f"[finetune] saved -> {save_path}")


if __name__ == "__main__":
    main()
