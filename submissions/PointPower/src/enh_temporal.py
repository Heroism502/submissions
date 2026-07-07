"""Temporal consistency utilities for Enh refinement (CG-aligned displacement smooth)."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

from uvg_io import read_ply_xyz_rgb, write_ply_xyz_rgb

FRAME_RE = re.compile(r"_(\d{4})\.ply$", re.IGNORECASE)
SmoothMode = Literal["mean", "ema"]


def parse_frame_id(path: str) -> int:
    match = FRAME_RE.search(path)
    if not match:
        raise ValueError(f"Cannot parse frame id from {path}")
    return int(match.group(1))


def cg_path_for_enh_ply(enh_ply: str, cg_root: str) -> str:
    """Map refine ENH ply path -> official CG ply under UVG-CWI-DQPC layout."""
    seq = os.path.basename(os.path.dirname(enh_ply))
    fname = os.path.basename(enh_ply).replace("_ENH_", "_CG_")
    return os.path.join(
        cg_root,
        seq,
        "consumer-grade_capture_system",
        "CG",
        "15fps",
        fname,
    )


def resample_model_to_ref(ref_xyz: np.ndarray, model_xyz: np.ndarray) -> np.ndarray:
    """NN-resample model_xyz onto ref_xyz point count / ordering."""
    if model_xyz.shape[0] == ref_xyz.shape[0]:
        return model_xyz.astype(np.float32)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(model_xyz)
    _, idx = nn.kneighbors(ref_xyz, return_distance=True)
    return model_xyz[idx[:, 0]].astype(np.float32)


def nearest_indices(src_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(src_xyz)
    _, idx = nn.kneighbors(query_xyz, return_distance=True)
    return idx[:, 0]


def transfer_ref_field_to_model(
    ref_xyz: np.ndarray,
    field_on_ref: np.ndarray,
    model_xyz: np.ndarray,
) -> np.ndarray:
    """Map per-ref values onto model points via nearest ref neighbor."""
    idx = nearest_indices(ref_xyz, model_xyz)
    return field_on_ref[idx]


def smooth_sequence_paths(
    paths: List[str],
    cg_paths: List[str],
    *,
    mode: SmoothMode = "mean",
    window: int = 5,
    ema_alpha: float = 0.35,
    max_correction_mm: float = 10.0,
) -> Tuple[List[np.ndarray], dict]:
    """Smooth ENH displacements from CG via cross-frame CG correspondence."""
    if not paths:
        return [], {"frames_smoothed": 0, "frames_fallback": 0, "max_correction_mm": 0.0}
    if len(paths) != len(cg_paths):
        raise ValueError("paths and cg_paths length mismatch")

    half = max(window // 2, 0)
    out: List[np.ndarray] = []
    frames_smoothed = 0
    frames_fallback = 0
    max_corr_seen = 0.0

    for i in range(len(paths)):
        lo = max(0, i - half)
        hi = min(len(paths), i + half + 1)
        cg_path = cg_paths[i]
        if not os.path.isfile(cg_path):
            xyz, _ = read_ply_xyz_rgb(paths[i])
            out.append(xyz.astype(np.float32))
            frames_fallback += 1
            continue

        ref_cg, _ = read_ply_xyz_rgb(cg_path)
        enh_i, _ = read_ply_xyz_rgb(paths[i])

        displacements: List[np.ndarray] = []
        ok = True
        for j in range(lo, hi):
            if not os.path.isfile(cg_paths[j]):
                ok = False
                break
            try:
                cg_j, _ = read_ply_xyz_rgb(cg_paths[j])
                enh_j, _ = read_ply_xyz_rgb(paths[j])
            except Exception:
                ok = False
                break
            enh_on_cg_j = resample_model_to_ref(cg_j, enh_j)
            idx_j = nearest_indices(cg_j, ref_cg)
            displacements.append(enh_on_cg_j[idx_j] - ref_cg)

        if not ok or not displacements:
            out.append(enh_i.astype(np.float32))
            frames_fallback += 1
            continue

        if mode == "ema":
            smoothed_disp = displacements[0].copy()
            a = float(np.clip(ema_alpha, 0.05, 1.0))
            for t in range(1, len(displacements)):
                smoothed_disp = a * displacements[t] + (1.0 - a) * smoothed_disp
        else:
            smoothed_disp = np.stack(displacements, axis=0).mean(axis=0)

        target_on_ref = ref_cg + smoothed_disp
        enh_i_on_ref = resample_model_to_ref(ref_cg, enh_i)
        correction_on_ref = target_on_ref - enh_i_on_ref
        if max_correction_mm > 0:
            mag = np.linalg.norm(correction_on_ref, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_correction_mm / np.maximum(mag, 1e-6))
            correction_on_ref = correction_on_ref * scale

        corr_on_enh = transfer_ref_field_to_model(ref_cg, correction_on_ref, enh_i)
        smoothed_enh = (enh_i + corr_on_enh).astype(np.float32)

        max_corr_seen = max(max_corr_seen, float(np.linalg.norm(corr_on_enh, axis=1).max()))
        if float(np.abs(corr_on_enh).max()) > 1e-6:
            frames_smoothed += 1
        out.append(smoothed_enh)

    return out, {
        "frames_smoothed": frames_smoothed,
        "frames_fallback": frames_fallback,
        "max_correction_mm": max_corr_seen,
    }


def collect_sequence_plys(in_dir: str, sequences: Optional[Iterable[str]] = None) -> Dict[str, List[Tuple[int, str]]]:
    by_seq: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for seq_name in sorted(os.listdir(in_dir)):
        seq_path = os.path.join(in_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue
        if sequences and seq_name not in sequences:
            continue
        for fname in os.listdir(seq_path):
            if not fname.endswith(".ply"):
                continue
            full = os.path.join(seq_path, fname)
            by_seq[seq_name].append((parse_frame_id(full), full))
    for seq in by_seq:
        by_seq[seq].sort(key=lambda x: x[0])
    return by_seq


def apply_temporal_smooth_dir(
    in_dir: str,
    out_dir: str,
    *,
    cg_root: str,
    window: int = 5,
    mode: SmoothMode = "mean",
    ema_alpha: float = 0.35,
    max_correction_mm: float = 10.0,
    sequences: Optional[Iterable[str]] = None,
) -> dict:
    """Smooth all sequences under in_dir/Seq/*.ply -> out_dir/Seq/*.ply."""
    by_seq = collect_sequence_plys(in_dir, sequences)
    stats = {
        "sequences": {},
        "frames_in": 0,
        "frames_out": 0,
        "frames_smoothed": 0,
        "frames_fallback": 0,
        "skipped_variable_topology": 0,
        "alignment": "cg_displacement",
        "cg_root": cg_root,
    }

    for seq_name, items in sorted(by_seq.items()):
        paths = [p for _, p in items]
        cg_paths = [cg_path_for_enh_ply(p, cg_root) for p in paths]
        out_seq_dir = os.path.join(out_dir, seq_name)
        os.makedirs(out_seq_dir, exist_ok=True)
        stats["frames_in"] += len(paths)

        smoothed, seq_stats = smooth_sequence_paths(
            paths,
            cg_paths,
            mode=mode,
            window=window,
            ema_alpha=ema_alpha,
            max_correction_mm=max_correction_mm,
        )
        stats["sequences"][seq_name] = {
            "smoothed": True,
            "num_frames": len(paths),
            **seq_stats,
        }
        stats["frames_smoothed"] += int(seq_stats["frames_smoothed"])
        stats["frames_fallback"] += int(seq_stats["frames_fallback"])

        for src_path, xyz_s in zip(paths, smoothed):
            _, rgb = read_ply_xyz_rgb(src_path)
            write_ply_xyz_rgb(os.path.join(out_seq_dir, os.path.basename(src_path)), xyz_s, rgb)
            stats["frames_out"] += 1

    stats["mode"] = mode
    stats["window"] = window
    stats["ema_alpha"] = ema_alpha
    stats["max_correction_mm"] = max_correction_mm
    # Legacy field: previously counted topology skips; now always 0 with CG alignment.
    stats["skipped_variable_topology"] = int(stats["frames_fallback"])
    return stats
