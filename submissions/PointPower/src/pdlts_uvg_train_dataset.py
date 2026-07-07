"""UVG CG→HE pair dataset for PD-LTS fine-tune (train split only).

Training pair in HE alignment frame (official metric convention):
  pcl_noisy = align(CG, M_{seq})
  pcl_clean = HE
"""
from __future__ import annotations

import os
import random
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import torch
from pytorch3d.ops import knn_points
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from gc2026_paths import resolve_gc2026_root  # noqa: E402

GC2026_ROOT = resolve_gc2026_root(SCRIPT_DIR)
METRIC_MATRICES = os.path.join(GC2026_ROOT, "code", "Metric", "matrices")

import sys

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from uvg_io import read_ply_xyz  # noqa: E402


def parse_pairs(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def sequence_from_path(path: str) -> str:
    marker = "/UVG-CWI-DQPC/"
    if marker in path:
        return path.split(marker, 1)[1].split("/")[0]
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path)))))


@lru_cache(maxsize=32)
def load_align_matrix(sequence: str) -> np.ndarray:
    p = os.path.join(METRIC_MATRICES, f"{sequence}.txt")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing alignment matrix: {p}")
    return np.loadtxt(p, dtype=np.float64).reshape(4, 4)


def apply_transform(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = xyz.astype(np.float64, copy=False)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (hom @ matrix.T)[:, :3].astype(np.float32)


def normalize_unit_sphere(pcl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_max = pcl.max(dim=0, keepdim=True)[0]
    p_min = pcl.min(dim=0, keepdim=True)[0]
    center = (p_max + p_min) / 2
    pcl = pcl - center
    scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0].clamp(min=1e-8)
    pcl = pcl / scale
    return pcl, center, scale


def make_patch_pair(
    pcl_noisy: torch.Tensor,
    pcl_clean: torch.Tensor,
    patch_size: int,
    seed_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = min(pcl_noisy.shape[0], pcl_clean.shape[0])
    if n < patch_size:
        raise ValueError(f"Too few points ({n}) for patch_size={patch_size}")
    if seed_idx is None:
        seed_idx = int(torch.randint(0, n, (1,)).item())
    seed = pcl_noisy[seed_idx : seed_idx + 1].unsqueeze(0)
    _, idx_a, pat_a = knn_points(seed, pcl_noisy.unsqueeze(0), K=patch_size, return_nn=True)
    idx_a = idx_a[0, 0]
    pat_b = pcl_clean[idx_a]
    return pat_a[0, 0], pat_b, seed[0, 0]


class UvgCgHeFrameDataset(Dataset):
    """One sample = one aligned CG/HE frame (subsampled)."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        max_points: int = 30_000,
        seed: int = 0,
    ):
        self.pairs = pairs
        self.max_points = max_points
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        cg_path, he_path = self.pairs[idx]
        seq = sequence_from_path(cg_path)
        M = load_align_matrix(seq)
        cg = read_ply_xyz(cg_path, max_points=self.max_points, rng=self.rng)
        he = read_ply_xyz(he_path, max_points=self.max_points, rng=self.rng)
        cg_aligned = apply_transform(cg, M)
        pcl_noisy = torch.from_numpy(cg_aligned)
        pcl_clean = torch.from_numpy(he.astype(np.float32, copy=False))
        pcl_clean, center, scale = normalize_unit_sphere(pcl_clean)
        pcl_noisy = (pcl_noisy - center) / scale
        return {
            "pcl_noisy": pcl_noisy,
            "pcl_clean": pcl_clean,
            "sequence": seq,
            "cg_path": cg_path,
        }


class UvgPatchDataset(Dataset):
    """Random patches from UVG CG/HE pairs (real noise, no synthetic AddNoise)."""

    def __init__(
        self,
        frame_dset: UvgCgHeFrameDataset,
        patch_size: int = 1024,
        patches_per_epoch: int = 2000,
    ):
        self.frame_dset = frame_dset
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, idx: int) -> dict:
        frame = self.frame_dset[idx % len(self.frame_dset)]
        pat_n, pat_c, seed = make_patch_pair(
            frame["pcl_noisy"], frame["pcl_clean"], self.patch_size,
        )
        return {
            "pcl_noisy": pat_n,
            "pcl_clean": pat_c,
            "seed_pnts": seed.unsqueeze(0),
        }


def build_train_loader(
    pairs_file: str,
    *,
    max_frames: int = 0,
    max_points: int = 30_000,
    patch_size: int = 1024,
    patches_per_epoch: int = 4000,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    pairs = parse_pairs(pairs_file)
    if max_frames > 0:
        pairs = pairs[:max_frames]
    if not pairs:
        raise ValueError(f"No training pairs in {pairs_file}")
    frame_dset = UvgCgHeFrameDataset(pairs, max_points=max_points)
    patch_dset = UvgPatchDataset(frame_dset, patch_size=patch_size, patches_per_epoch=patches_per_epoch)
    return DataLoader(
        patch_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(patch_dset) >= batch_size,
    )


def summarize_train_pairs(pairs_file: str, max_frames: int = 0) -> dict:
    pairs = parse_pairs(pairs_file)
    if max_frames > 0:
        pairs = pairs[:max_frames]
    seqs = sorted({sequence_from_path(cg) for cg, _ in pairs})
    missing_mats = [s for s in seqs if not os.path.isfile(os.path.join(METRIC_MATRICES, f"{s}.txt"))]
    return {
        "num_pairs": len(pairs),
        "sequences": seqs,
        "missing_matrices": missing_mats,
    }
