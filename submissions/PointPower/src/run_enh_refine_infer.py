#!/usr/bin/env python3
"""Batch multi-stage Enh refinement: pre/post SOR, PD-LTS, snap, fill, adaptive route."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from gc2026_paths import resolve_gc2026_root  # noqa: E402

GC2026_ROOT = resolve_gc2026_root(SCRIPT_DIR)

from enh_refine_config import (  # noqa: E402
    GRID_PRESETS,
    PHASE2A_PRESETS,
    PHASE2B_PRESETS,
    PHASE2C_PRESETS,
    ROLLBACK_CONFIG,
    RefineConfig,
    resolve_preset,
)
from enh_geometry_sources import pick_geometry_dir, pick_secondary_geometry_dir, source_status  # noqa: E402
from enh_refine_pipeline import (
    apply_refine_stages,
    estimate_cg_inlier_ratio,
    output_ply_path,
    process_cg_frame,
    sequence_from_cg_path,
)  # noqa: E402
from enh_temporal_attention import build_enh_history_by_cg
from enh_temporal_region import (  # noqa: E402
    build_sequence_frame_index,
    neighbor_cg_paths_for_frame,
    neighbor_frames_for_frame,
)


def needs_temporal_neighbors(cfg: RefineConfig) -> bool:
    extra = cfg.extra or {}
    return extra.get("hybrid_fill_role") in (
        "temporal_region_mask",
        "temporal_attn_region_mask",
    )


def temporal_half_window_for(cfg: RefineConfig) -> int:
    tw = int(cfg.temporal_window or (cfg.extra or {}).get("temporal_window", 5))
    return max(1, tw // 2)


def load_pdlts_denoise(model: str, device: str):
    import run_pdlts_infer as pdlts  # noqa: WPS433

    get_net, large_patch, denoise_loop = pdlts.load_denoise_module(model)
    ckpt = pdlts.default_ckpt(model)
    network = get_net(ckpt).to(device)
    network.eval()
    cluster_size = 50000
    large_threshold = 50000

    def _denoise(xyz):
        return pdlts.denoise_xyz(
            xyz,
            network,
            large_patch,
            denoise_loop,
            cluster_size=cluster_size,
            large_threshold=large_threshold,
            device=device,
            verbose=False,
        )

    return _denoise


def load_config(args) -> RefineConfig:
    if args.config_json:
        return RefineConfig.from_json_file(args.config_json)
    if args.preset:
        cfg = resolve_preset(args.preset)
    elif args.refine_config:
        with open(args.refine_config, encoding="utf-8") as f:
            data = json.load(f)
        if "production_config" in data:
            data = data["production_config"]
        elif "best_config" in data:
            data = data["best_config"]
        cfg = RefineConfig.from_dict(data)
    else:
        raise SystemExit("Provide --preset, --config-json, or --refine-config")
    if args.geometry_dir:
        cfg.geometry_dir = args.geometry_dir
        if cfg.geometry.startswith("pdlts_") and args.use_geometry_cache:
            cfg.geometry = "from_dir"
            cfg.extra = {**(cfg.extra or {}), "geometry_source": f"pdlts_{cfg.pdlts_model}"}
    elif cfg.geometry == "from_dir" or (
        (cfg.extra or {}).get("geometry_source") and cfg.geometry not in ("hybrid_pdlts_superpc",)
    ):
        gdir, label = pick_geometry_dir(cfg, root=GC2026_ROOT)
        if gdir:
            cfg.geometry_dir = gdir
            if cfg.geometry != "hybrid_pdlts_superpc":
                cfg.geometry = "from_dir"
    if cfg.geometry == "hybrid_pdlts_superpc":
        if args.geometry_secondary_dir:
            cfg.extra = {**(cfg.extra or {}), "geometry_secondary_dir": args.geometry_secondary_dir}
        else:
            sec_dir, _ = pick_secondary_geometry_dir(cfg, "", GC2026_ROOT)
            if sec_dir:
                cfg.extra = {**(cfg.extra or {}), "geometry_secondary_dir": sec_dir}
    return cfg


def apply_geometry_cache_cfg(frame_cfg: RefineConfig, args) -> RefineConfig:
    use_cache = args.use_geometry_cache or args.require_geometry_cache or frame_cfg.geometry == "from_dir"
    if not use_cache or frame_cfg.geometry == "passthrough_cg":
        return frame_cfg
    out = RefineConfig.from_dict(frame_cfg.to_dict())
    if frame_cfg.geometry in ("pdlts_light", "pdlts_heavy"):
        src = out.geometry
        out.extra = {**(out.extra or {}), "geometry_source": src}
        out.geometry = "from_dir"
    if out.geometry in ("from_dir", "hybrid_pdlts_superpc"):
        gdir, _ = pick_geometry_dir(out, args.geometry_dir, GC2026_ROOT)
        if gdir:
            out.geometry_dir = gdir
        if out.geometry == "hybrid_pdlts_superpc":
            if args.geometry_secondary_dir:
                out.extra = {**(out.extra or {}), "geometry_secondary_dir": args.geometry_secondary_dir}
            else:
                sec_dir, sec_label = pick_secondary_geometry_dir(out, "", GC2026_ROOT)
                if sec_dir:
                    out.extra = {**(out.extra or {}), "geometry_secondary_dir": sec_dir}
                    out.extra["geometry_secondary_resolved"] = sec_label
            out.geometry = "hybrid_pdlts_superpc"
        elif frame_cfg.geometry in ("pdlts_light", "pdlts_heavy"):
            out.geometry = "from_dir"
    elif frame_cfg.geometry == "from_dir" and not frame_cfg.geometry_dir:
        gdir, _ = pick_geometry_dir(frame_cfg, args.geometry_dir, GC2026_ROOT)
        if gdir:
            out = RefineConfig.from_dict(frame_cfg.to_dict())
            out.geometry_dir = gdir
            return out
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Multi-stage Enh refinement batch infer")
    p.add_argument("--cg-list", required=True)
    p.add_argument("--out-dir", default=os.path.join(GC2026_ROOT, "output", "enh_refine"))
    p.add_argument("--preset", default="", help=f"Named preset: {', '.join(p.name for p in GRID_PRESETS[:5])}...")
    p.add_argument("--config-json", default="", help="Full RefineConfig JSON")
    p.add_argument(
        "--refine-config",
        default="",
        help="Gate decision JSON (uses production_config or best_config)",
    )
    p.add_argument(
        "--geometry-dir",
        default="",
        help="Cached PD-LTS / geometry root (Seq/ENH.ply layout)",
    )
    p.add_argument(
        "--geometry-secondary-dir",
        default="",
        help="Hybrid preset: SuperPC secondary cache (overrides geometry_secondary source)",
    )
    p.add_argument(
        "--use-geometry-cache",
        action="store_true",
        help="With --geometry-dir: skip live PD-LTS, read cached geometry",
    )
    p.add_argument(
        "--require-geometry-cache",
        action="store_true",
        help="Never run live PD-LTS; missing cache uses geometry-fallback",
    )
    p.add_argument(
        "--geometry-fallback",
        default="filter_cg",
        choices=["filter_cg", "passthrough_cg", "skip"],
        help="When geometry cache missing for a frame",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--save-config", action="store_true", default=True)
    p.add_argument("--no-save-config", action="store_false", dest="save_config")
    p.add_argument(
        "--per-seq-config",
        default="",
        help="JSON with sequences.{name} -> RefineConfig overrides",
    )
    p.add_argument(
        "--enh-history-dir",
        default="",
        help="Prior refine ENH cache for temporal-attention mask (Seq/ENH.ply layout)",
    )
    p.add_argument(
        "--frame-proxy-json",
        default="",
        help="Proxy rollback rules (from build_per_frame_refine_decision.py)",
    )
    return p.parse_args()


def frame_proxy_passthrough(cg_path: str, cg_xyz, cg_rgb, proxy: dict) -> bool:
    seq = sequence_from_cg_path(cg_path)
    ir = estimate_cg_inlier_ratio(cg_xyz, cg_rgb)
    n_pts = int(cg_xyz.shape[0])
    for rule in proxy.get("rules", []):
        if rule.get("sequence") == seq and rule.get("action") == "passthrough":
            if ir >= float(rule.get("min_inlier_ratio", 1.0)) and n_pts >= int(rule.get("min_points", 0)):
                return True
    gr = proxy.get("global_rule", {})
    if gr and gr.get("action") == "passthrough":
        if ir >= float(gr.get("min_inlier_ratio", 1.0)) and n_pts >= int(gr.get("min_points", 0)):
            return True
    return False


def main():
    args = parse_args()
    cfg = load_config(args)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.save_config:
        with open(os.path.join(args.out_dir, "pipeline_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    with open(args.cg_list, "r", encoding="utf-8") as f:
        cg_paths = [ln.strip().split("\t")[0] for ln in f if ln.strip() and not ln.startswith("#")]
    if args.max_samples > 0:
        cg_paths = cg_paths[: args.max_samples]

    sequence_index: dict = {}
    temporal_half_window = 0
    enh_by_cg: dict = {}
    if needs_temporal_neighbors(cfg):
        sequence_index = build_sequence_frame_index(cg_paths)
        temporal_half_window = temporal_half_window_for(cfg)
        if args.enh_history_dir:
            enh_by_cg = build_enh_history_by_cg(args.enh_history_dir, cg_paths)
            print(
                f"[refine] enh history: {len(enh_by_cg)} frames from {args.enh_history_dir}"
            )
        print(
            f"[refine] temporal index: {len(sequence_index)} sequences, "
            f"half_window={temporal_half_window}"
        )

    denoise_fn = None
    needs_pdlts = cfg.geometry in ("pdlts_light", "pdlts_heavy")
    use_cache = (
        args.use_geometry_cache
        or args.require_geometry_cache
        or cfg.geometry in ("from_dir", "hybrid_pdlts_superpc")
    )
    if needs_pdlts and use_cache:
        src = cfg.geometry if cfg.geometry in ("pdlts_light", "pdlts_heavy") else f"pdlts_{cfg.pdlts_model}"
        cfg.extra = {**(cfg.extra or {}), "geometry_source": src}
        cfg.geometry = "from_dir"
        gdir, _ = pick_geometry_dir(cfg, args.geometry_dir, GC2026_ROOT)
        if gdir:
            cfg.geometry_dir = gdir
        needs_pdlts = False
    if needs_pdlts:
        if args.require_geometry_cache:
            raise SystemExit(
                "Preset requires live PD-LTS but --require-geometry-cache set. "
                "Wait for Phase-1 or use a from_dir preset."
            )
        model = cfg.pdlts_model or ("light" if cfg.geometry == "pdlts_light" else "heavy")
        device = args.device if torch.cuda.is_available() else "cpu"
        denoise_fn = load_pdlts_denoise(model, device)
        print(f"[refine] loaded PD-LTS model={model} on {device}")
    elif cfg.geometry in ("from_dir", "hybrid_pdlts_superpc"):
        gdir, label = pick_geometry_dir(cfg, args.geometry_dir, GC2026_ROOT)
        if gdir:
            cfg.geometry_dir = gdir
            print(f"[refine] geometry source={label} dir={gdir}")
        elif args.require_geometry_cache:
            raise SystemExit(f"No geometry cache for preset {cfg.name}")
        if cfg.geometry == "hybrid_pdlts_superpc":
            if args.geometry_secondary_dir:
                cfg.extra = {**(cfg.extra or {}), "geometry_secondary_dir": args.geometry_secondary_dir}
                print(f"[refine] hybrid secondary=cli dir={args.geometry_secondary_dir}")
            else:
                sec_dir, sec_label = pick_secondary_geometry_dir(cfg, "", GC2026_ROOT)
                if sec_dir:
                    cfg.extra = {**(cfg.extra or {}), "geometry_secondary_dir": sec_dir}
                    print(f"[refine] hybrid secondary={sec_label} dir={sec_dir}")

    records = []
    skipped = 0
    started = datetime.utcnow().isoformat() + "Z"
    t_all = time.perf_counter()

    per_seq_map: dict = {}
    per_seq_default: Optional[RefineConfig] = None
    if args.per_seq_config and os.path.isfile(args.per_seq_config):
        with open(args.per_seq_config, encoding="utf-8") as f:
            ps = json.load(f)
        per_seq_default = RefineConfig.from_dict(ps.get("default", cfg.to_dict()))
        per_seq_map = ps.get("sequences", {})

    frame_proxy: dict = {}
    if args.frame_proxy_json and os.path.isfile(args.frame_proxy_json):
        with open(args.frame_proxy_json, encoding="utf-8") as f:
            frame_proxy = json.load(f)

    from uvg_io import read_ply_xyz_rgb  # noqa: WPS433

    for cg_path in tqdm(cg_paths, desc=f"refine:{cfg.tag()}"):
        if not os.path.isfile(cg_path):
            print(f"[WARN] missing {cg_path}")
            continue
        out_path = output_ply_path(args.out_dir, cg_path)
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue
        frame_cfg = cfg
        seq = sequence_from_cg_path(cg_path)
        if seq in per_seq_map:
            frame_cfg = RefineConfig.from_dict({**cfg.to_dict(), **per_seq_map[seq]})
        elif per_seq_default is not None:
            frame_cfg = RefineConfig.from_dict({**cfg.to_dict(), **per_seq_default.to_dict()})
        frame_cfg = apply_geometry_cache_cfg(frame_cfg, args)
        t0 = time.perf_counter()
        gdir, _ = pick_geometry_dir(frame_cfg, args.geometry_dir, GC2026_ROOT)
        neighbor_paths: Optional[list] = None
        neighbor_frames = None
        if needs_temporal_neighbors(frame_cfg) and sequence_index:
            fill_role = (frame_cfg.extra or {}).get("hybrid_fill_role", "")
            if fill_role == "temporal_attn_region_mask":
                neighbor_frames = neighbor_frames_for_frame(
                    cg_path, sequence_index, temporal_half_window, enh_by_cg=enh_by_cg,
                )
            else:
                neighbor_paths = neighbor_cg_paths_for_frame(
                    cg_path, sequence_index, temporal_half_window,
                )
        try:
            if frame_proxy:
                cg_xyz, cg_rgb = read_ply_xyz_rgb(cg_path)
                if frame_proxy_passthrough(cg_path, cg_xyz, cg_rgb, frame_proxy):
                    frame_cfg = ROLLBACK_CONFIG
                    gdir = ""
            meta = process_cg_frame(
                cg_path,
                out_path,
                frame_cfg,
                denoise_fn=denoise_fn,
                geometry_dir=gdir or args.geometry_dir or frame_cfg.geometry_dir,
                geometry_fallback=args.geometry_fallback,
                neighbor_cg_paths=neighbor_paths,
                neighbor_frames=neighbor_frames,
            )
            meta["seconds"] = round(time.perf_counter() - t0, 3)
            records.append(meta)
        except Exception as exc:
            if args.geometry_fallback == "skip":
                print(f"[WARN] skip {cg_path}: {exc}")
                continue
            print(f"[WARN] {cg_path}: {exc} -> rollback passthrough")
            meta = process_cg_frame(
                cg_path,
                out_path,
                ROLLBACK_CONFIG,
                geometry_dir="",
                geometry_fallback="passthrough_cg",
            )
            meta["seconds"] = round(time.perf_counter() - t0, 3)
            meta["fallback"] = str(exc)
            records.append(meta)

    finished = datetime.utcnow().isoformat() + "Z"
    summary = {
        "started": started,
        "finished": finished,
        "config": cfg.to_dict(),
        "processed": len(records),
        "skipped_existing": skipped,
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "records": records,
    }
    with open(os.path.join(args.out_dir, "infer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done {len(records)} frames ({skipped} skipped) -> {args.out_dir}")


if __name__ == "__main__":
    main()
