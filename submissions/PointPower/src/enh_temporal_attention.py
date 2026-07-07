"""Multi-frame temporal attention for hybrid SuperPC fill masking.

Combines:
  - CG correspondence over a wide temporal window (softmax attention per point)
  - Optional ENH history (prior refine pass) for displacement agreement
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from enh_temporal import resample_model_to_ref
from uvg_io import read_ply_xyz_rgb


@dataclass(frozen=True)
class TemporalNeighbor:
    frame_id: int
    delta_frames: int
    cg_path: str
    enh_path: Optional[str] = None


def _softmax_over_neighbors(logits: np.ndarray) -> np.ndarray:
    """Softmax along axis 0 for shape (J, N)."""
    m = np.max(logits, axis=0, keepdims=True)
    exp = np.exp(logits - m)
    return exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-8)


def build_enh_history_by_cg(enh_history_dir: str, cg_paths: List[str]) -> Dict[str, str]:
    """Map official CG path -> matching ENH ply under history dir (if present)."""
    from enh_refine_pipeline import output_ply_path

    out: Dict[str, str] = {}
    if not enh_history_dir:
        return out
    for cg in cg_paths:
        enh = output_ply_path(enh_history_dir, cg)
        if os.path.isfile(enh):
            out[cg] = enh
    return out


def compute_attention_temporal_stability(
    ref_cg_xyz: np.ndarray,
    neighbors: List[TemporalNeighbor],
    *,
    match_mm: float = 15.0,
    agree_mm: float = 8.0,
    temporal_tau: float = 8.0,
    dist_tau: float = 12.0,
    cg_weight: float = 0.35,
    enh_weight: float = 0.65,
    max_neighbor_points: int = 120_000,
) -> Tuple[np.ndarray, dict]:
    """
    Per-CG-point stability in [0, 1] via softmax attention over temporal neighbors.

    CG branch: weighted presence of correspondences within match_mm.
    ENH branch (when history available): low displacement variance across neighbors.
    """
    from sklearn.neighbors import NearestNeighbors

    n = ref_cg_xyz.shape[0]
    if not neighbors:
        return np.ones(n, dtype=np.float32), {"temporal_attn_neighbors": 0, "temporal_mode": "no_neighbors"}

    j_count = len(neighbors)
    rng = np.random.RandomState(0)
    logits = np.full((j_count, n), -1e9, dtype=np.float32)
    cg_match = np.zeros((j_count, n), dtype=np.float32)
    disp_stack = np.full((j_count, n, 3), np.nan, dtype=np.float32)
    valid_enh = 0

    for j, nb in enumerate(neighbors):
        if not os.path.isfile(nb.cg_path):
            continue
        cg_j, _ = read_ply_xyz_rgb(nb.cg_path, max_points=max_neighbor_points, rng=rng)
        if cg_j.shape[0] == 0:
            continue
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(cg_j)
        dist, idx = nn.kneighbors(ref_cg_xyz, return_distance=True)
        dist = dist[:, 0]
        idx = idx[:, 0]
        matched = dist < float(match_mm)
        cg_match[j] = matched.astype(np.float32)
        time_w = np.exp(-abs(nb.delta_frames) / max(float(temporal_tau), 1e-3))
        spatial = np.exp(-dist / max(float(dist_tau), 1e-3))
        logits[j] = np.log(max(time_w, 1e-6)) + np.log(np.maximum(spatial, 1e-9))
        logits[j, ~matched] = -1e9

        if nb.enh_path and os.path.isfile(nb.enh_path):
            try:
                enh_j, _ = read_ply_xyz_rgb(nb.enh_path, max_points=max_neighbor_points, rng=rng)
                enh_on_cg_j = resample_model_to_ref(cg_j, enh_j)
                disp_stack[j] = (enh_on_cg_j[idx] - ref_cg_xyz).astype(np.float32)
                valid_enh += 1
            except Exception:
                pass

    attn = _softmax_over_neighbors(logits)
    cg_stab = (attn * cg_match).sum(axis=0).astype(np.float32)

    if valid_enh == 0:
        stability = cg_stab
        meta = {
            "temporal_attn_neighbors": j_count,
            "temporal_attn_enh_neighbors": 0,
            "temporal_attn_mode": "cg_only",
            "temporal_attn_mean_stability": float(stability.mean()),
            "temporal_attn_frac_interior": float((stability >= 0.6).mean()),
        }
        return stability, meta

    has_disp = ~np.isnan(disp_stack[:, :, 0])
    attn_enh = attn * has_disp.astype(np.float32)
    w_sum = np.maximum(attn_enh.sum(axis=0), 1e-8)
    disp_mean = np.nan_to_num(attn_enh[:, :, None] * disp_stack, nan=0.0).sum(axis=0) / w_sum[:, None]
    err = np.linalg.norm(np.nan_to_num(disp_stack, nan=0.0) - disp_mean[None, :, :], axis=2)
    local_agree = np.exp(-err / max(float(agree_mm), 1e-3))
    enh_stab = (attn_enh * local_agree).sum(axis=0) / w_sum

    tot_w = max(float(cg_weight) + float(enh_weight), 1e-6)
    stability = (
        float(cg_weight) * cg_stab + float(enh_weight) * enh_stab.astype(np.float32)
    ) / tot_w

    meta = {
        "temporal_attn_neighbors": j_count,
        "temporal_attn_enh_neighbors": valid_enh,
        "temporal_attn_mode": "cg+enh",
        "temporal_match_mm": float(match_mm),
        "temporal_attn_tau_time": float(temporal_tau),
        "temporal_attn_tau_dist": float(dist_tau),
        "temporal_disp_agree_mm": float(agree_mm),
        "temporal_attn_mean_stability": float(stability.mean()),
        "temporal_attn_frac_interior": float((stability >= 0.6).mean()),
    }
    return stability.astype(np.float32), meta
