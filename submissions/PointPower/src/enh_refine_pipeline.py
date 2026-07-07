"""Apply multi-stage Enh refinement on a single CG frame."""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from enh_temporal_attention import TemporalNeighbor

from enh_refine_config import FILTER_CG_CONFIG, ROLLBACK_CONFIG, RefineConfig
from frame_fill_gate import apply_tier_to_config, decide_frame_fill_gate, tier_fill_overrides
from uvg_io import (
    filter_cg_outliers,
    merge_cg_model_fill,
    merge_cg_model_fill_density_adaptive,
    merge_primary_fill_cg_holes,
    merge_primary_fill_cg_holes_density_adaptive,
    merge_xyz_rgb_voxel,
    read_ply_xyz_rgb,
    snap_bidirectional_cg_model,
    snap_xyz_to_reference,
    transfer_colors_knn,
    write_ply_xyz_rgb,
)


def sequence_from_cg_path(cg_path: str) -> str:
    marker = "/UVG-CWI-DQPC/"
    if marker in cg_path:
        return cg_path.split(marker, 1)[1].split("/")[0]
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cg_path)))))


def output_ply_path(out_dir: str, cg_path: str) -> str:
    seq = sequence_from_cg_path(cg_path)
    fname = os.path.basename(cg_path)
    out_name = fname.replace("_CG_", "_ENH_", 1)
    return os.path.join(out_dir, seq, out_name)


def geometry_ply_path(geometry_dir: str, cg_path: str) -> str:
    return output_ply_path(geometry_dir, cg_path)


def filter_superpc_by_cg_region(
    cg_xyz: np.ndarray,
    secondary_xyz: np.ndarray,
    secondary_rgb: np.ndarray,
    r_in_mm: float,
    r_out_mm: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Keep SuperPC points only in CG interior (d<r_in); discard exterior (d>=r_out)."""
    from sklearn.neighbors import NearestNeighbors

    if secondary_xyz.shape[0] == 0:
        return secondary_xyz, secondary_rgb, {"region_interior_kept": 0, "region_discarded": 0}
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(cg_xyz)
    dist, _ = nn.kneighbors(secondary_xyz, return_distance=True)
    dist = dist[:, 0]
    interior = dist < float(r_in_mm)
    if r_out_mm > r_in_mm:
        interior &= dist < float(r_out_mm)
    meta = {
        "region_r_in_mm": float(r_in_mm),
        "region_r_out_mm": float(r_out_mm),
        "region_interior_kept": int(interior.sum()),
        "region_discarded": int((~interior).sum()),
        "region_superpc_total": int(secondary_xyz.shape[0]),
    }
    return secondary_xyz[interior], secondary_rgb[interior], meta


def estimate_cg_inlier_ratio(xyz: np.ndarray, rgb: np.ndarray) -> float:
    try:
        filtered_xyz, _ = filter_cg_outliers(xyz, rgb, nb_neighbors=20, std_ratio=2.0)
        return float(filtered_xyz.shape[0]) / max(int(xyz.shape[0]), 1)
    except Exception:
        return 1.0


def adaptive_snap_mm(
    cfg: RefineConfig,
    cg_xyz: np.ndarray,
    cg_rgb: np.ndarray,
    model_xyz: Optional[np.ndarray] = None,
) -> float:
    """Rule-based snap: return effective snap_mm (0 = skip snap, like VH tune)."""
    extra = cfg.extra or {}
    if not extra.get("adaptive_snap"):
        return cfg.snap_mm
    rule = extra.get("adaptive_snap_rule", "inlier")
    if rule == "inlier":
        thresh = float(extra.get("adaptive_snap_inlier_min", 0.97))
        ratio = estimate_cg_inlier_ratio(cg_xyz, cg_rgb)
        if ratio >= thresh:
            return 0.0
        return cfg.snap_mm
    if rule == "geometry_close" and model_xyz is not None and model_xyz.shape[0] > 0:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(model_xyz)
        dist, _ = nn.kneighbors(cg_xyz, return_distance=True)
        med = float(np.median(dist))
        thresh = float(extra.get("adaptive_snap_geom_median_mm", 0.8))
        if med <= thresh:
            return 0.0
        return cfg.snap_mm
    return cfg.snap_mm


def adaptive_geometry_mode(cfg: RefineConfig, xyz: np.ndarray, rgb: np.ndarray) -> RefineConfig:
    """Per-frame routing: very clean CG -> passthrough; noisy -> keep refine stages."""
    if not cfg.adaptive_route:
        return cfg
    ratio = estimate_cg_inlier_ratio(xyz, rgb)
    n_pts = int(xyz.shape[0])
    out = RefineConfig(**cfg.to_dict())
    if ratio > 0.985 and n_pts > 520000:
        out.geometry = "passthrough_cg"
        out.fill_mm = 0.0
        out.blend_voxel_mm = 0.0
        out.name = cfg.name + "_routed_pass"
        return out
    if ratio > 0.96:
        out.fill_mm = min(out.fill_mm, 0.6)
        out.snap_mm = max(out.snap_mm, 0.5)
        out.name = cfg.name + "_routed_mild"
        return out
    out.name = cfg.name + "_routed_aggressive"
    return out


def resample_geometry_to_cg(
    cg_xyz: np.ndarray,
    cg_rgb: np.ndarray,
    model_xyz: np.ndarray,
    model_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fix topology to CG point count (for temporal smooth / stable point count)."""
    if model_xyz.shape[0] == cg_xyz.shape[0]:
        return model_xyz, model_rgb
    out_xyz = snap_xyz_to_reference(cg_xyz, model_xyz, snap_mm=1e9)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(model_xyz)
    _, idx = nn.kneighbors(cg_xyz, return_distance=True)
    out_rgb = model_rgb[idx[:, 0]]
    # Use nearest model point coords (not cg shell) for geometry
    out_xyz = model_xyz[idx[:, 0]].astype(np.float32)
    return out_xyz, out_rgb.astype(np.float32)


def _fill_anchor(
    cfg: RefineConfig,
    ref_xyz: np.ndarray,
    ref_rgb: np.ndarray,
    out_xyz: np.ndarray,
    out_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    extra = cfg.extra or {}
    if cfg.geometry == "hybrid_pdlts_superpc" and extra.get("hybrid_fill_anchor", "cg") == "primary":
        return out_xyz, out_rgb
    return ref_xyz, ref_rgb


def _apply_primary_density_refine(
    ref_xyz: np.ndarray,
    ref_rgb: np.ndarray,
    out_xyz: np.ndarray,
    out_rgb: np.ndarray,
    pdr: dict,
    meta: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Always-on ft-style snap + density fill on primary (before optional SuperPC)."""
    snap_mm = float(pdr.get("snap_mm", 0))
    fill_mm = float(pdr.get("fill_mm", 0))
    if snap_mm > 0:
        out_xyz = snap_xyz_to_reference(out_xyz, ref_xyz, snap_mm)
        meta["primary_density_snap_mm"] = snap_mm
    if fill_mm > 0:
        fill_mode = pdr.get("fill_mode", "density_adaptive")
        k = int(pdr.get("fill_density_k", 6))
        scale_max = float(pdr.get("fill_density_scale_max", 2.0))
        before = int(out_xyz.shape[0])
        if fill_mode == "density_adaptive":
            out_xyz, out_rgb = merge_cg_model_fill_density_adaptive(
                ref_xyz,
                ref_rgb,
                out_xyz,
                out_rgb,
                fill_mm,
                k_neighbors=k,
                scale_max=scale_max,
            )
        else:
            out_xyz, out_rgb = merge_cg_model_fill(
                ref_xyz, ref_rgb, out_xyz, out_rgb, fill_mm,
            )
        meta["primary_density_fill_mm"] = fill_mm
        meta["primary_density_added"] = int(out_xyz.shape[0] - before)
    meta["primary_density_refine"] = True
    meta["primary_density_points"] = int(out_xyz.shape[0])
    return out_xyz, out_rgb


def _run_merge_fill(
    cfg: RefineConfig,
    anchor_xyz: np.ndarray,
    anchor_rgb: np.ndarray,
    fill_model_xyz: np.ndarray,
    fill_model_rgb: np.ndarray,
    cg_xyz: np.ndarray,
    meta: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    extra = cfg.extra or {}
    hole_mask = extra.get("hybrid_hole_mask", "anchor")
    max_fill_ratio = float(extra.get("hybrid_max_fill_ratio", 0.0) or 0.0)
    if hole_mask == "cg_holes" and cfg.geometry == "hybrid_pdlts_superpc":
        if cfg.fill_mode == "density_adaptive":
            out_xyz, out_rgb = merge_primary_fill_cg_holes_density_adaptive(
                anchor_xyz,
                anchor_rgb,
                fill_model_xyz,
                fill_model_rgb,
                cg_xyz,
                cfg.fill_mm,
                k_neighbors=cfg.fill_density_k,
                scale_max=cfg.fill_density_scale_max,
                max_fill_ratio=max_fill_ratio,
            )
            meta["fill_mode"] = "density_adaptive_cg_holes"
        else:
            out_xyz, out_rgb = merge_primary_fill_cg_holes(
                anchor_xyz,
                anchor_rgb,
                fill_model_xyz,
                fill_model_rgb,
                cg_xyz,
                cfg.fill_mm,
                max_fill_ratio=max_fill_ratio,
            )
            meta["fill_mode"] = "cg_holes"
        meta["hybrid_hole_mask"] = "cg_holes"
        if max_fill_ratio > 0:
            meta["hybrid_max_fill_ratio"] = max_fill_ratio
            meta["hybrid_fill_added"] = int(max(0, out_xyz.shape[0] - anchor_xyz.shape[0]))
        return out_xyz, out_rgb

    if cfg.fill_mode == "density_adaptive":
        out_xyz, out_rgb = merge_cg_model_fill_density_adaptive(
            anchor_xyz,
            anchor_rgb,
            fill_model_xyz,
            fill_model_rgb,
            cfg.fill_mm,
            k_neighbors=cfg.fill_density_k,
            scale_max=cfg.fill_density_scale_max,
        )
        meta["fill_mode"] = "density_adaptive"
    else:
        out_xyz, out_rgb = merge_cg_model_fill(
            anchor_xyz, anchor_rgb, fill_model_xyz, fill_model_rgb, cfg.fill_mm,
        )
    return out_xyz, out_rgb


def apply_refine_stages(
    cg_xyz: np.ndarray,
    cg_rgb: np.ndarray,
    cfg: RefineConfig,
    *,
    cg_path: str = "",
    denoise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    geometry_dir: str = "",
    geometry_fallback: str = "filter_cg",
    neighbor_cg_paths: Optional[List[str]] = None,
    neighbor_frames: Optional[List[TemporalNeighbor]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Run configured stages. Always keeps original CG as reference anchor."""
    meta = {"config_name": cfg.name, "geometry": cfg.geometry}
    cfg = adaptive_geometry_mode(cfg, cg_xyz, cg_rgb)
    meta["effective_geometry"] = cfg.geometry

    work_xyz, work_rgb = cg_xyz, cg_rgb
    if cfg.pre_sor:
        work_xyz, work_rgb = filter_cg_outliers(
            work_xyz, work_rgb, nb_neighbors=cfg.pre_sor_nb, std_ratio=cfg.pre_sor_std,
        )
        meta["pre_sor_points"] = int(work_xyz.shape[0])

    ref_xyz, ref_rgb = cg_xyz, cg_rgb

    fill_model_xyz: Optional[np.ndarray] = None
    fill_model_rgb: Optional[np.ndarray] = None

    if cfg.geometry == "passthrough_cg":
        out_xyz, out_rgb = ref_xyz, ref_rgb
    elif cfg.geometry == "filter_cg":
        out_xyz, out_rgb = filter_cg_outliers(ref_xyz, ref_rgb, nb_neighbors=20, std_ratio=2.0)
    elif cfg.geometry in ("pdlts_light", "pdlts_heavy"):
        if denoise_fn is None:
            raise ValueError(f"geometry={cfg.geometry} requires denoise_fn")
        model_xyz = denoise_fn(work_xyz)
        model_rgb = transfer_colors_knn(ref_xyz, ref_rgb, model_xyz, k=1)
        out_xyz, out_rgb = model_xyz, model_rgb
        meta["model_points"] = int(model_xyz.shape[0])
    elif cfg.geometry in ("from_dir", "hybrid_pdlts_superpc"):
        gdir = geometry_dir or cfg.geometry_dir
        if not gdir:
            raise ValueError(f"geometry={cfg.geometry} requires geometry_dir")
        if not cg_path:
            raise ValueError(f"geometry={cfg.geometry} requires cg_path")
        gpath = geometry_ply_path(gdir, cg_path)
        if not os.path.isfile(gpath):
            meta["geometry_miss"] = True
            meta["geometry_miss_path"] = gpath
            fb = geometry_fallback or cfg.geometry_fallback or "filter_cg"
            if fb == "skip":
                raise FileNotFoundError(f"Missing geometry ply: {gpath}")
            fb_cfg = FILTER_CG_CONFIG if fb == "filter_cg" else ROLLBACK_CONFIG
            meta["geometry_fallback"] = fb_cfg.name
            return apply_refine_stages(
                cg_xyz, cg_rgb, fb_cfg, cg_path=cg_path, geometry_fallback=geometry_fallback,
            )
        primary_xyz, primary_rgb = read_ply_xyz_rgb(gpath)
        meta["geometry_path"] = gpath
        out_xyz, out_rgb = primary_xyz, primary_rgb

        if cfg.geometry == "hybrid_pdlts_superpc":
            sec_dir = (cfg.extra or {}).get("geometry_secondary_dir", "")
            fill_role = (cfg.extra or {}).get("hybrid_fill_role", "union_voxel")
            voxel_mm = float((cfg.extra or {}).get("hybrid_voxel_mm", 0.5))
            meta["hybrid_fill_role"] = fill_role
            if sec_dir:
                sec_path = geometry_ply_path(sec_dir, cg_path)
                if os.path.isfile(sec_path):
                    secondary_xyz, secondary_rgb = read_ply_xyz_rgb(sec_path)
                    meta["geometry_secondary_path"] = sec_path
                    meta["hybrid_primary_points"] = int(primary_xyz.shape[0])
                    meta["hybrid_secondary_points"] = int(secondary_xyz.shape[0])
                    if fill_role == "secondary_only":
                        out_xyz, out_rgb = primary_xyz, primary_rgb
                        fill_model_xyz, fill_model_rgb = secondary_xyz, secondary_rgb
                        meta["hybrid_surface"] = "primary"
                        meta["hybrid_fill_source"] = "secondary"
                    elif fill_role == "region_mask":
                        r_in = float((cfg.extra or {}).get("region_r_in_mm", 25.0))
                        r_out = float((cfg.extra or {}).get("region_r_out_mm", 45.0))
                        out_xyz, out_rgb = primary_xyz, primary_rgb
                        fill_model_xyz, fill_model_rgb, rmeta = filter_superpc_by_cg_region(
                            ref_xyz, secondary_xyz, secondary_rgb, r_in, r_out,
                        )
                        meta.update(rmeta)
                        meta["hybrid_surface"] = "primary_pdlts"
                    elif fill_role == "temporal_region_mask":
                        from enh_temporal_region import (
                            compute_cg_temporal_stability,
                            filter_superpc_by_temporal_region,
                        )

                        extra = cfg.extra or {}
                        stability, smeta = compute_cg_temporal_stability(
                            ref_xyz,
                            neighbor_cg_paths or [],
                            match_mm=float(extra.get("temporal_match_mm", 15.0)),
                        )
                        out_xyz, out_rgb = primary_xyz, primary_rgb
                        fill_model_xyz, fill_model_rgb, rmeta = filter_superpc_by_temporal_region(
                            ref_xyz,
                            stability,
                            secondary_xyz,
                            secondary_rgb,
                            tau_interior=float(extra.get("temporal_tau_interior", 0.6)),
                            tau_exterior=float(extra.get("temporal_tau_exterior", 0.2)),
                            cg_link_mm=float(extra.get("temporal_cg_link_mm", 25.0)),
                        )
                        meta.update(smeta)
                        meta.update(rmeta)
                        meta["hybrid_surface"] = "primary_pdlts_temporal"
                    elif fill_role == "temporal_attn_region_mask":
                        from enh_temporal_attention import compute_attention_temporal_stability
                        from enh_temporal_region import filter_superpc_by_temporal_region

                        extra = cfg.extra or {}
                        nb_frames = neighbor_frames or []
                        stability, smeta = compute_attention_temporal_stability(
                            ref_xyz,
                            nb_frames,
                            match_mm=float(extra.get("temporal_match_mm", 15.0)),
                            agree_mm=float(extra.get("temporal_disp_agree_mm", 8.0)),
                            temporal_tau=float(extra.get("temporal_attn_tau_time", 8.0)),
                            dist_tau=float(extra.get("temporal_attn_tau_dist", 12.0)),
                            cg_weight=float(extra.get("temporal_cg_weight", 0.35)),
                            enh_weight=float(extra.get("temporal_enh_weight", 0.65)),
                        )
                        out_xyz, out_rgb = primary_xyz, primary_rgb
                        fill_model_xyz, fill_model_rgb, rmeta = filter_superpc_by_temporal_region(
                            ref_xyz,
                            stability,
                            secondary_xyz,
                            secondary_rgb,
                            tau_interior=float(extra.get("temporal_tau_interior", 0.6)),
                            tau_exterior=float(extra.get("temporal_tau_exterior", 0.2)),
                            cg_link_mm=float(extra.get("temporal_cg_link_mm", 25.0)),
                        )
                        meta.update(smeta)
                        meta.update(rmeta)
                        meta["hybrid_surface"] = "primary_pdlts_temporal_attn"
                    else:
                        out_xyz, out_rgb = merge_xyz_rgb_voxel(
                            [primary_xyz, secondary_xyz],
                            [primary_rgb, secondary_rgb],
                            voxel_mm,
                        )
                        fill_model_xyz, fill_model_rgb = out_xyz, out_rgb
                        meta["hybrid_surface"] = "union_voxel"
                        meta["hybrid_fused_points"] = int(out_xyz.shape[0])
                else:
                    meta["geometry_secondary_miss"] = True
                    meta["geometry_secondary_miss_path"] = sec_path
            else:
                meta["geometry_secondary_miss"] = True
    else:
        raise ValueError(f"Unknown geometry mode: {cfg.geometry}")

    if fill_model_xyz is None:
        fill_model_xyz, fill_model_rgb = out_xyz, out_rgb

    if (cfg.extra or {}).get("fp_filter_before_fill") and cfg.fill_mm > 0:
        ref_xyz, ref_rgb = filter_cg_outliers(
            ref_xyz, ref_rgb, nb_neighbors=cfg.pre_sor_nb, std_ratio=cfg.pre_sor_std,
        )
        meta["fp_filter_before_fill"] = True

    extra = cfg.extra or {}
    pdr = extra.get("primary_density_refine")
    if pdr and cfg.geometry == "hybrid_pdlts_superpc":
        out_xyz, out_rgb = _apply_primary_density_refine(
            ref_xyz, ref_rgb, out_xyz, out_rgb, pdr, meta,
        )

    fill_before_snap = bool(extra.get("fill_before_snap"))
    anchor_xyz, anchor_rgb = _fill_anchor(cfg, ref_xyz, ref_rgb, out_xyz, out_rgb)
    if extra.get("hybrid_fill_anchor") == "primary":
        meta["hybrid_fill_anchor"] = "primary"

    can_snap = cfg.snap_mm > 0 and cfg.geometry not in ("passthrough_cg", "filter_cg")
    can_fill = cfg.fill_mm > 0 and cfg.geometry not in ("passthrough_cg", "filter_cg")

    def run_snap() -> None:
        nonlocal out_xyz, ref_xyz, fill_model_xyz
        if not can_snap:
            return
        effective_snap = adaptive_snap_mm(cfg, ref_xyz, ref_rgb, out_xyz)
        if effective_snap > 0:
            if cfg.bidirectional_snap:
                out_xyz, ref_xyz = snap_bidirectional_cg_model(
                    out_xyz, ref_xyz, effective_snap, cfg.cg_pull_mm,
                )
                meta["bidirectional_snap"] = True
                meta["cg_pull_mm"] = cfg.cg_pull_mm
            else:
                out_xyz = snap_xyz_to_reference(out_xyz, ref_xyz, effective_snap)
            if not fill_before_snap:
                fill_model_xyz = snap_xyz_to_reference(fill_model_xyz, ref_xyz, effective_snap)
            meta["snap_mm"] = effective_snap
        else:
            meta["snap_mm"] = 0.0
            meta["adaptive_snap_skip"] = True

    def run_fill() -> None:
        nonlocal out_xyz, out_rgb, anchor_xyz, anchor_rgb
        if not can_fill:
            return
        fill_cfg = cfg
        extra_gate = extra.get("frame_fill_gate")
        if extra_gate:
            skip_seqs = extra.get("frame_fill_gate_skip_sequences") or []
            seq = sequence_from_cg_path(cg_path) if cg_path else ""
            if seq in skip_seqs:
                tier = "skip"
                gate_metrics = {"frame_fill_gate_forced_skip_sequence": seq}
                meta["frame_fill_gate"] = tier
                meta.update(gate_metrics)
                meta["frame_fill_gate_post_sor"] = False
                meta["superpc_fill_skipped"] = "sequence_skip"
                return
            tier, gate_metrics = decide_frame_fill_gate(
                ref_xyz, out_xyz, extra, fill_model_xyz,
            )
            meta["frame_fill_gate"] = tier
            meta.update(gate_metrics)
            overrides = tier_fill_overrides(extra, tier)
            eff_fill = float(overrides.get("fill_mm", cfg.fill_mm))
            meta["frame_fill_gate_post_sor"] = bool(overrides.get("post_sor", cfg.post_sor))
            if eff_fill <= 0:
                meta["superpc_fill_skipped"] = "frame_fill_gate_skip"
                return
            fill_cfg = apply_tier_to_config(cfg, overrides)
            meta["frame_fill_gate_fill_mm"] = float(fill_cfg.fill_mm)
            if "hybrid_max_fill_ratio" in overrides:
                meta["frame_fill_gate_max_fill_ratio"] = float(overrides["hybrid_max_fill_ratio"])
        anchor_xyz, anchor_rgb = _fill_anchor(fill_cfg, ref_xyz, ref_rgb, out_xyz, out_rgb)
        out_xyz, out_rgb = _run_merge_fill(
            fill_cfg, anchor_xyz, anchor_rgb, fill_model_xyz, fill_model_rgb, ref_xyz, meta,
        )
        meta["fill_mm"] = fill_cfg.fill_mm

    anchor_before = int(out_xyz.shape[0])
    if fill_before_snap:
        meta["stage_order"] = "fill_then_snap"
        run_fill()
        run_snap()
    else:
        meta["stage_order"] = "snap_then_fill"
        run_snap()
        run_fill()

    if cfg.blend_voxel_mm > 0:
        out_xyz, out_rgb = merge_xyz_rgb_voxel(
            [ref_xyz, out_xyz], [ref_rgb, out_rgb], cfg.blend_voxel_mm,
        )
        meta["blend_voxel_mm"] = cfg.blend_voxel_mm

    if cfg.post_sor:
        skip_post = False
        if "frame_fill_gate_post_sor" in meta and not meta["frame_fill_gate_post_sor"]:
            skip_post = True
            meta["post_sor_skipped"] = "frame_fill_gate_tier"
        if extra.get("adaptive_post_sor") and not skip_post:
            added_ratio = max(0, out_xyz.shape[0] - anchor_before) / max(anchor_before, 1)
            min_ratio = float(extra.get("adaptive_post_sor_min_add_ratio", 0.02))
            if added_ratio < min_ratio:
                skip_post = True
                meta["post_sor_skipped"] = "low_superpc_fill"
                meta["post_sor_skip_add_ratio"] = float(added_ratio)
        if not skip_post:
            out_xyz, out_rgb = filter_cg_outliers(
                out_xyz, out_rgb, nb_neighbors=cfg.post_sor_nb, std_ratio=cfg.post_sor_std,
            )
            meta["post_sor_points"] = int(out_xyz.shape[0])

    return out_xyz.astype(np.float32), out_rgb.astype(np.float32), meta


def process_cg_frame(
    cg_path: str,
    out_path: str,
    cfg: RefineConfig,
    *,
    denoise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    geometry_dir: str = "",
    geometry_fallback: str = "filter_cg",
    neighbor_cg_paths: Optional[List[str]] = None,
    neighbor_frames: Optional[List[TemporalNeighbor]] = None,
) -> dict:
    cg_xyz, cg_rgb = read_ply_xyz_rgb(cg_path)
    meta = {"cg_path": cg_path, "out_path": out_path}
    out_xyz, out_rgb, stage_meta = apply_refine_stages(
        cg_xyz,
        cg_rgb,
        cfg,
        cg_path=cg_path,
        denoise_fn=denoise_fn,
        geometry_dir=geometry_dir or cfg.geometry_dir,
        geometry_fallback=geometry_fallback,
        neighbor_cg_paths=neighbor_cg_paths,
        neighbor_frames=neighbor_frames,
    )
    meta.update(stage_meta)
    meta["input_points"] = int(cg_xyz.shape[0])
    meta["output_points"] = int(out_xyz.shape[0])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_ply_xyz_rgb(out_path, out_xyz, out_rgb)
    return meta
