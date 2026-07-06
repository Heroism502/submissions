#!/usr/bin/env python3
"""Batch PD-LTS denoising on UVG CG PLY frames (Enhancement path).

Reads colored CG PLY, denoises XYZ with PD-LTS (clustered patch inference for
large clouds), transfers RGB from input via KNN, writes ENH PLY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from gc2026_paths import resolve_gc2026_root  # noqa: E402

GC2026_ROOT = resolve_gc2026_root(SCRIPT_DIR)
PDLTS_ROOT = os.environ.get("PDLTS_ROOT", "").strip() or os.path.join(GC2026_ROOT, "code", "PD-LTS")
from uvg_io import read_ply_xyz_rgb, transfer_colors_knn, write_ply_xyz_rgb  # noqa: E402


def sequence_from_cg_path(cg_path: str) -> str:
    marker = "/UVG-CWI-DQPC/"
    if marker in cg_path:
        return cg_path.split(marker, 1)[1].split("/")[0]
    parts = cg_path.replace("\\", "/").split("/")
    for root_name in ("full_pipeline_cg", "full_pipeline_val_cg"):
        if root_name in parts:
            idx = parts.index(root_name)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cg_path)))))


def output_ply_path(out_dir: str, cg_path: str) -> str:
    seq = sequence_from_cg_path(cg_path)
    fname = os.path.basename(cg_path)
    out_name = fname.replace("_CG_", "_ENH_", 1)
    return os.path.join(out_dir, seq, out_name)


def setup_pdlts_runtime() -> None:
    """Patch imports so PD-LTS loads for inference without kaolin / full training deps."""
    from pytorch3d.loss import chamfer_distance as p3d_cd

    kaolin = types.ModuleType("kaolin")
    km = types.ModuleType("kaolin.metrics")
    kpc = types.ModuleType("kaolin.metrics.pointcloud")
    kpc.chamfer_distance = p3d_cd
    km.pointcloud = kpc
    kaolin.metrics = km
    for name, mod in (
        ("kaolin", kaolin),
        ("kaolin.metrics", km),
        ("kaolin.metrics.pointcloud", kpc),
    ):
        sys.modules[name] = mod

    _orig_load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load_compat  # type: ignore[method-assign]

    if PDLTS_ROOT not in sys.path:
        sys.path.insert(0, PDLTS_ROOT)
    os.chdir(PDLTS_ROOT)


def default_ckpt(model: str) -> str:
    if model == "heavy":
        return os.path.join(PDLTS_ROOT, "product", "ckpt", "Denoiseflow-heavy-FBM.ckpt")
    return os.path.join(PDLTS_ROOT, "product", "ckpt", "Denoiseflow-light-FBM.ckpt")


def load_denoise_module(model: str):
    setup_pdlts_runtime()
    if model == "heavy":
        from models.model_heavy import denoise as denoise_mod  # noqa: WPS433
    else:
        from models.model_light import denoise as denoise_mod  # noqa: WPS433
    return denoise_mod.get_denoise_net, denoise_mod.large_patch_denoise_v1, denoise_mod.denoise_loop


def denoise_xyz(
    xyz: np.ndarray,
    network,
    large_patch_denoise_v1,
    denoise_loop,
    *,
    cluster_size: int,
    large_threshold: int,
    device: str,
    verbose: bool,
) -> np.ndarray:
    pcl = torch.from_numpy(xyz.astype(np.float32, copy=False))
    if pcl.shape[0] <= large_threshold:
        from dataset.scoredenoise.transforms import NormalizeUnitSphere  # noqa: WPS433

        class _Args:
            patch_size = 1000
            seed_k = 3
            niters = 1
            device = device

        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl)
        pcl_next = denoise_loop(_Args(), network, pcl_noisy, seed_k_alpha=pcl.shape[0] / 10000.0)
        out = (pcl_next * scale + center).numpy()
    else:
        out_t = large_patch_denoise_v1(
            network,
            pcl,
            cluster_size=cluster_size,
            device=device,
            verbose=verbose,
        )
        out = out_t.detach().cpu().numpy()
    return out.astype(np.float32, copy=False)


def parse_args():
    p = argparse.ArgumentParser(description="Batch PD-LTS denoise for UVG CG PLY")
    p.add_argument("--cg-list", required=True, help="One CG ply path per line")
    p.add_argument(
        "--out-dir",
        default=os.path.join(GC2026_ROOT, "output", "pdlts_enhanced"),
    )
    p.add_argument(
        "--ckpt",
        default="",
        help="PD-LTS checkpoint (.ckpt); default from --model",
    )
    p.add_argument("--model", choices=["light", "heavy"], default="light")
    p.add_argument("--cluster-size", type=int, default=50000)
    p.add_argument(
        "--large-threshold",
        type=int,
        default=50000,
        help="Use clustered denoise when point count exceeds this",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all frames")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--color-knn", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = args.ckpt or default_ckpt(args.model)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"PD-LTS checkpoint not found: {ckpt}")

    get_denoise_net, large_patch_denoise_v1, denoise_loop = load_denoise_module(args.model)
    device = args.device if torch.cuda.is_available() else "cpu"
    network = get_denoise_net(ckpt).to(device)
    network.eval()

    with open(args.cg_list, "r", encoding="utf-8") as f:
        cg_paths = [line.strip().split("\t")[0] for line in f if line.strip() and not line.startswith("#")]
    if args.max_samples > 0:
        cg_paths = cg_paths[: args.max_samples]

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "runtime.log")
    started = datetime.utcnow().isoformat() + "Z"
    records = []
    skipped = 0

    for cg_path in tqdm(cg_paths, desc="PD-LTS infer"):
        if not os.path.isfile(cg_path):
            print(f"[WARN] missing: {cg_path}")
            continue

        out_path = output_ply_path(args.out_dir, cg_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue

        t0 = time.perf_counter()
        xyz, rgb = read_ply_xyz_rgb(cg_path)
        out_xyz = denoise_xyz(
            xyz,
            network,
            large_patch_denoise_v1,
            denoise_loop,
            cluster_size=args.cluster_size,
            large_threshold=args.large_threshold,
            device=device,
            verbose=args.verbose,
        )
        out_rgb = transfer_colors_knn(xyz, rgb, out_xyz, k=args.color_knn)
        write_ply_xyz_rgb(out_path, out_xyz, out_rgb)

        elapsed = time.perf_counter() - t0
        records.append(
            {
                "cg_path": cg_path,
                "out_path": out_path,
                "input_points": int(xyz.shape[0]),
                "output_points": int(out_xyz.shape[0]),
                "model": args.model,
                "cluster_size": args.cluster_size,
                "seconds": round(elapsed, 3),
            }
        )

    finished = datetime.utcnow().isoformat() + "Z"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"start={started}\n")
        f.write(f"end={finished}\n")
        f.write(f"ckpt={ckpt}\n")
        f.write(f"model={args.model}\n")
        f.write(f"cluster_size={args.cluster_size}\n")
        f.write(f"processed={len(records)}\n")
        f.write(f"skipped_existing={skipped}\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta_path = os.path.join(args.out_dir, "infer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"started": started, "finished": finished, "records": records}, f, indent=2)

    print(f"Done. {len(records)} frames ({skipped} skipped) -> {args.out_dir}")
    if records:
        avg = sum(r["seconds"] for r in records) / len(records)
        print(f"[summary] avg_sec={avg:.2f} | first={records[0]['out_path']}")


if __name__ == "__main__":
    main()
