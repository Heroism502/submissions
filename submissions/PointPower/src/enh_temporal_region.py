"""Temporal-consistency region masks for hybrid PD-LTS + SuperPC refine."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from enh_temporal import parse_frame_id
from enh_temporal_attention import TemporalNeighbor
from uvg_io import read_ply_xyz_rgb


def neighbor_cg_paths_for_frame(
    cg_path: str,
    sequence_index: dict,
    half_window: int,
) -> List[str]:
    """Return CG paths of temporal neighbors (same sequence, |Δframe|<=half_window)."""
    return [nb.cg_path for nb in neighbor_frames_for_frame(cg_path, sequence_index, half_window)]


def neighbor_frames_for_frame(
    cg_path: str,
    sequence_index: dict,
    half_window: int,
    enh_by_cg: Optional[Dict[str, str]] = None,
) -> List[TemporalNeighbor]:
    """Temporal neighbors with optional ENH history paths keyed by CG path."""
    from enh_refine_pipeline import sequence_from_cg_path

    seq = sequence_from_cg_path(cg_path)
    fid = parse_frame_id(cg_path)
    frames = sequence_index.get(seq, [])
    out: List[TemporalNeighbor] = []
    enh_map = enh_by_cg or {}
    for nf, path in frames:
        if nf == fid:
            continue
        if abs(nf - fid) <= half_window:
            out.append(
                TemporalNeighbor(
                    frame_id=nf,
                    delta_frames=nf - fid,
                    cg_path=path,
                    enh_path=enh_map.get(path),
                )
            )
    return out


def build_sequence_frame_index(cg_paths: List[str]) -> dict:
    """sequence -> sorted list of (frame_id, cg_path)."""
    from collections import defaultdict

    from enh_refine_pipeline import sequence_from_cg_path

    by_seq: dict = defaultdict(list)
    for path in cg_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            by_seq[sequence_from_cg_path(path)].append((parse_frame_id(path), path))
        except ValueError:
            continue
    for seq in by_seq:
        by_seq[seq].sort(key=lambda x: x[0])
    return dict(by_seq)


def compute_cg_temporal_stability(
    cg_xyz: np.ndarray,
    neighbor_cg_paths: List[str],
    match_mm: float = 15.0,
    max_neighbor_points: int = 120_000,
) -> Tuple[np.ndarray, dict]:
    """Per-CG-point stability in [0,1]: fraction of neighbor frames with a nearby point."""
    from sklearn.neighbors import NearestNeighbors

    n = cg_xyz.shape[0]
    if not neighbor_cg_paths:
        return np.ones(n, dtype=np.float32), {"temporal_neighbors": 0, "temporal_mode": "no_neighbors"}

    rng = np.random.RandomState(0)
    stable = np.zeros(n, dtype=np.float32)
    valid = 0
    for npath in neighbor_cg_paths:
        if not os.path.isfile(npath):
            continue
        n_xyz, _ = read_ply_xyz_rgb(npath, max_points=max_neighbor_points, rng=rng)
        if n_xyz.shape[0] == 0:
            continue
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(n_xyz)
        dist, _ = nn.kneighbors(cg_xyz, return_distance=True)
        stable += (dist[:, 0] < float(match_mm)).astype(np.float32)
        valid += 1

    if valid == 0:
        return np.ones(n, dtype=np.float32), {"temporal_neighbors": 0, "temporal_mode": "missing_neighbors"}

    stability = stable / float(valid)
    meta = {
        "temporal_neighbors": valid,
        "temporal_match_mm": float(match_mm),
        "temporal_mean_stability": float(stability.mean()),
        "temporal_frac_interior": float((stability >= 0.6).mean()),
    }
    return stability, meta


def classify_cg_regions(
    stability: np.ndarray,
    tau_interior: float,
    tau_exterior: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks: interior, boundary, exterior."""
    interior = stability >= float(tau_interior)
    exterior = stability < float(tau_exterior)
    boundary = (~interior) & (~exterior)
    return interior, boundary, exterior


def filter_superpc_by_temporal_region(
    cg_xyz: np.ndarray,
    stability: np.ndarray,
    secondary_xyz: np.ndarray,
    secondary_rgb: np.ndarray,
    tau_interior: float = 0.6,
    tau_exterior: float = 0.2,
    cg_link_mm: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Interior (temporally stable CG): allow SuperPC fill.
    Exterior / boundary: discard SuperPC (surface stays PD-LTS).
    """
    from sklearn.neighbors import NearestNeighbors

    interior_cg, boundary_cg, exterior_cg = classify_cg_regions(
        stability, tau_interior, tau_exterior,
    )
    meta = {
        "temporal_tau_interior": float(tau_interior),
        "temporal_tau_exterior": float(tau_exterior),
        "temporal_cg_link_mm": float(cg_link_mm),
        "temporal_cg_interior_frac": float(interior_cg.mean()),
        "temporal_cg_boundary_frac": float(boundary_cg.mean()),
        "temporal_cg_exterior_frac": float(exterior_cg.mean()),
    }
    if secondary_xyz.shape[0] == 0:
        meta.update({"region_interior_kept": 0, "region_discarded": 0})
        return secondary_xyz, secondary_rgb, meta

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(cg_xyz)
    dist, idx = nn.kneighbors(secondary_xyz, return_distance=True)
    dist = dist[:, 0]
    idx = idx[:, 0]
    st_at_sp = stability[idx]
    keep = (st_at_sp >= tau_interior) & (dist < float(cg_link_mm))
    meta.update(
        {
            "region_interior_kept": int(keep.sum()),
            "region_discarded": int((~keep).sum()),
            "region_superpc_total": int(secondary_xyz.shape[0]),
            "hybrid_fill_source": "superpc_temporal_interior",
        }
    )
    return secondary_xyz[keep], secondary_rgb[keep], meta
