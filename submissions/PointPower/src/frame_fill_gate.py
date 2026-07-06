"""Per-frame gate: skip / lite / full SuperPC CG-hole fill from estimated fill benefit."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from enh_refine_config import RefineConfig
from uvg_io import estimate_primary_fill_add_ratio


def compute_cg_spacing_stats(cg_xyz: np.ndarray, k_neighbors: int = 6) -> Tuple[float, float]:
    from sklearn.neighbors import NearestNeighbors

    if cg_xyz.shape[0] < 2:
        return 0.0, 0.0
    k = min(max(int(k_neighbors), 2), cg_xyz.shape[0])
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(cg_xyz)
    dists, _ = nn.kneighbors(cg_xyz, return_distance=True)
    local = dists[:, -1]
    pos = local[local > 0]
    if pos.size == 0:
        return 0.0, 0.0
    return float(np.median(pos)), float(np.percentile(pos, 90))


def decide_frame_fill_gate(
    cg_xyz: np.ndarray,
    primary_xyz: np.ndarray,
    extra: Dict[str, Any],
    secondary_xyz: Optional[np.ndarray] = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Return (tier, metrics) where tier is skip | lite | full.

    Primary signal: estimated SuperPC add ratio under cg_holes mask (lite params).
    skip — negligible fill benefit (VictoryHeart-like); full — high benefit (sparse TS).
    """
    spacing_med, spacing_p90 = compute_cg_spacing_stats(
        cg_xyz, k_neighbors=int(extra.get("frame_fill_gate_k", 6)),
    )
    fill_mm = float(extra.get("frame_fill_gate_probe_fill_mm", 0.25))
    scale_max = float(extra.get("frame_fill_gate_probe_scale_max", 1.3))
    max_ratio = float(extra.get("frame_fill_gate_probe_max_ratio", 0.10))
    k_fill = int(extra.get("frame_fill_gate_probe_k", 6))

    est_ratio = 0.0
    if secondary_xyz is not None and secondary_xyz.shape[0] > 0:
        est_ratio = estimate_primary_fill_add_ratio(
            primary_xyz,
            secondary_xyz,
            cg_xyz,
            fill_mm,
            k_neighbors=k_fill,
            scale_max=scale_max,
            max_fill_ratio=max_ratio,
        )

    tau_skip = float(extra.get("frame_fill_gate_skip_add_ratio", 0.022))
    tau_full = float(extra.get("frame_fill_gate_full_add_ratio", 0.055))

    if est_ratio < tau_skip:
        tier = "skip"
    elif est_ratio >= tau_full:
        tier = "full"
    else:
        tier = "lite"

    metrics = {
        "frame_fill_gate_est_add_ratio": est_ratio,
        "frame_fill_gate_cg_spacing_med_mm": spacing_med,
        "frame_fill_gate_cg_spacing_p90_mm": spacing_p90,
    }
    return tier, metrics


def tier_fill_overrides(extra: Dict[str, Any], tier: str) -> Dict[str, Any]:
    tiers = extra.get("frame_fill_gate_tiers") or {}
    base = dict(tiers.get(tier, {}))
    if tier == "skip" and "fill_mm" not in base:
        base["fill_mm"] = 0.0
        base["post_sor"] = False
    return base


def apply_tier_to_config(cfg: RefineConfig, overrides: Dict[str, Any]) -> RefineConfig:
    d = cfg.to_dict()
    extra = dict(d.get("extra") or {})
    for key, val in overrides.items():
        if key in (
            "hybrid_max_fill_ratio",
            "adaptive_post_sor",
            "adaptive_post_sor_min_add_ratio",
        ):
            extra[key] = val
        elif key in d:
            d[key] = val
    d["extra"] = extra
    return RefineConfig(**d)
