#!/usr/bin/env python3
"""Evaluate point clouds with UVG-CWI GC baseline metric definitions.

Matches ACMMM26_GC_baseline.csv / code/Metric/cpp (evaluate_points):
  - Apply per-sequence alignment matrix (CG -> HE) to the *test* cloud only.
  - accuracy = mean NN dist test -> HE
  - completeness = mean NN dist HE -> test
  - chamfer_distance = 0.5 * (accuracy + completeness)
  - hausdorff_distance = max(max dist both directions)
  - precision/recall/fscore at thresholds use strict ``dist < t`` (mm)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from gc2026_paths import resolve_gc2026_root  # noqa: E402

GC2026_ROOT = resolve_gc2026_root(SCRIPT_DIR)
METRIC_MATRICES = os.path.join(GC2026_ROOT, "code", "Metric", "matrices")
DEFAULT_THRESHOLDS = (10.0, 20.0, 30.0, 50.0)

from evaluate_uvg import enh_path_from_cg  # noqa: E402
from uvg_io import cg_to_he_path, parse_frame_id, read_ply_xyz  # noqa: E402


@lru_cache(maxsize=32)
def load_align_matrix(sequence: str) -> np.ndarray:
    path = os.path.join(METRIC_MATRICES, f"{sequence}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing alignment matrix for sequence {sequence}: {path}")
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def apply_transform(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = xyz.astype(np.float64, copy=False)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (hom @ matrix.T)[:, :3]


def sequence_from_cg_path(cg_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cg_path)))))


def compute_gc_metrics(
    test_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, float]:
    if test_xyz.shape[0] == 0 or gt_xyz.shape[0] == 0:
        out: dict[str, float] = {
            "chamfer_distance": 0.0,
            "accuracy": 0.0,
            "completeness": 0.0,
            "hausdorff_distance": 0.0,
        }
        for t in thresholds:
            ts = f"{int(t)}.0"
            out[f"precision_{ts}"] = 0.0
            out[f"recall_{ts}"] = 0.0
            out[f"fscore_{ts}"] = 0.0
        return out

    # Avoid nested all-core parallelism when ProcessPoolExecutor runs many workers.
    kdtree_workers = 1 if os.environ.get("GC2026_EVAL_PARALLEL") == "1" else -1
    tree_gt = cKDTree(gt_xyz)
    tree_test = cKDTree(test_xyz)
    d_gt_to_test, _ = tree_test.query(gt_xyz, k=1, workers=kdtree_workers)
    d_test_to_gt, _ = tree_gt.query(test_xyz, k=1, workers=kdtree_workers)

    accuracy = float(d_test_to_gt.mean())
    completeness = float(d_gt_to_test.mean())
    chamfer = 0.5 * (accuracy + completeness)
    hausdorff = float(max(d_gt_to_test.max(), d_test_to_gt.max()))

    out = {
        "chamfer_distance": chamfer,
        "accuracy": accuracy,
        "completeness": completeness,
        "hausdorff_distance": hausdorff,
    }
    for t in thresholds:
        p = float((d_test_to_gt < t).mean())
        r = float((d_gt_to_test < t).mean())
        f = 0.0 if (p + r) == 0.0 else 2.0 * p * r / (p + r)
        ts = f"{int(t)}.0"
        out[f"precision_{ts}"] = p
        out[f"recall_{ts}"] = r
        out[f"fscore_{ts}"] = f
    return out


def eval_aligned_pair(
    test_path: str,
    gt_path: str,
    sequence: str,
    thresholds: tuple[float, ...],
    max_load_points: int,
    seed: int,
) -> dict:
    rng = np.random.RandomState(seed)
    test_xyz = read_ply_xyz(test_path, max_points=max_load_points, rng=rng)
    gt_xyz = read_ply_xyz(gt_path, max_points=max_load_points, rng=rng)
    test_xyz = apply_transform(test_xyz, load_align_matrix(sequence))
    metrics = compute_gc_metrics(test_xyz, gt_xyz, thresholds)
    metrics.update(
        {
            "test_file": os.path.basename(test_path),
            "gt_file": os.path.basename(gt_path),
            "sequence": sequence,
            "n_test": int(test_xyz.shape[0]),
            "n_gt": int(gt_xyz.shape[0]),
        }
    )
    return metrics


def _worker_task(payload: dict) -> dict | None:
    try:
        eval_args = {
            k: payload[k]
            for k in (
                "test_path",
                "gt_path",
                "sequence",
                "thresholds",
                "max_load_points",
                "seed",
            )
        }
        m = eval_aligned_pair(**eval_args)
        m["frame_id"] = payload["frame_id"]
        m["cg_path"] = payload.get("cg_path")
        m["test_path"] = payload["test_path"]
        m["gt_path"] = payload["gt_path"]

        if payload.get("also_cg") and payload.get("cg_path"):
            cg_args = dict(eval_args)
            cg_args["test_path"] = payload["cg_path"]
            m_cg = eval_aligned_pair(**cg_args)
            m["cg_chamfer_distance"] = m_cg["chamfer_distance"]
            m["cg_accuracy"] = m_cg["accuracy"]
            m["cg_completeness"] = m_cg["completeness"]
            m["delta_cg_minus_enh"] = m_cg["chamfer_distance"] - m["chamfer_distance"]
        return m
    except Exception as exc:  # noqa: BLE001
        return {"frame_id": payload.get("frame_id"), "error": str(exc)}


def parse_pairs_file(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            cg = parts[0]
            he = parts[1] if len(parts) > 1 and parts[1] else cg_to_he_path(cg)
            pairs.append((cg, he))
    return pairs


def aggregate_metrics(records: Iterable[dict], keys: list[str]) -> dict[str, float]:
    vals: dict[str, list[float]] = {k: [] for k in keys}
    for rec in records:
        if rec.get("error"):
            continue
        for k in keys:
            if k in rec and rec[k] is not None:
                vals[k].append(float(rec[k]))
    return {k: float(np.mean(v)) if v else None for k, v in vals.items()}


def metric_keys(thresholds: tuple[float, ...]) -> list[str]:
    keys = ["accuracy", "completeness", "hausdorff_distance", "chamfer_distance"]
    for t in thresholds:
        ts = f"{int(t)}.0"
        keys.extend([f"precision_{ts}", f"recall_{ts}", f"fscore_{ts}"])
    return keys


def write_csv(path: str, records: list[dict], thresholds: tuple[float, ...]) -> None:
    fieldnames = [
        "frame",
        "test_file",
        "gt_file",
        "chamfer_distance",
        "accuracy",
        "completeness",
        "hausdorff_distance",
    ]
    for t in thresholds:
        ts = f"{int(t)}.0"
        fieldnames.extend([f"precision_{ts}", f"recall_{ts}", f"fscore_{ts}"])
    fieldnames.append("sequence")

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, rec in enumerate(records):
            if rec.get("error"):
                continue
            row = dict(rec)
            row["frame"] = i
            w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GC baseline-compatible metric evaluation")
    p.add_argument(
        "--pairs-file",
        default=os.path.join(GC2026_ROOT, "data/processed/all_pairs_cgv2.txt"),
    )
    p.add_argument("--test-root", default=None, help="If set, evaluate ENH/CG under this root")
    p.add_argument(
        "--test-mode",
        choices=("enh", "cg"),
        default="enh",
        help="Resolve test PLY from CG path (enh or cg)",
    )
    p.add_argument("--thresholds", default="10,20,30,50")
    p.add_argument("--max-load-points", type=int, default=0, help="0 = load all points")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument(
        "--also-cg",
        action="store_true",
        help="Also score aligned CG vs HE (for improvement vs baseline)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())
    pairs = parse_pairs_file(args.pairs_file)
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]

    tasks: list[dict] = []
    for idx, (cg_path, he_path) in enumerate(pairs):
        seq = sequence_from_cg_path(cg_path)
        if args.test_mode == "enh":
            if not args.test_root:
                raise SystemExit("--test-root required for enh mode")
            test_path = enh_path_from_cg(cg_path, args.test_root)
        else:
            test_path = cg_path
        if not os.path.isfile(test_path) or not os.path.isfile(he_path):
            continue
        tasks.append(
            {
                "test_path": test_path,
                "gt_path": he_path,
                "sequence": seq,
                "thresholds": thresholds,
                "max_load_points": args.max_load_points,
                "seed": args.seed + idx,
                "frame_id": parse_frame_id(cg_path),
                "cg_path": cg_path,
                "also_cg": args.also_cg,
            }
        )

    records: list[dict] = []
    if args.workers <= 1:
        for payload in tqdm(tasks, desc="gc_metrics"):
            rec = _worker_task(payload)
            if rec:
                records.append(rec)
    else:
        os.environ["GC2026_EVAL_PARALLEL"] = "1"
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker_task, p) for p in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="gc_metrics"):
                rec = fut.result()
                if rec:
                    records.append(rec)

    records.sort(key=lambda r: (r.get("sequence", ""), r.get("frame_id", "")))
    keys = metric_keys(thresholds)
    ok = [r for r in records if not r.get("error")]
    cg_chamfer = [float(r["cg_chamfer_distance"]) for r in ok if "cg_chamfer_distance" in r]
    enh_chamfer = [float(r["chamfer_distance"]) for r in ok if "chamfer_distance" in r]
    deltas = [float(r["delta_cg_minus_enh"]) for r in ok if "delta_cg_minus_enh" in r]

    summary = {
        "eval_mode": "gc_baseline_aligned",
        "pairs_file": args.pairs_file,
        "test_mode": args.test_mode,
        "test_root": args.test_root,
        "thresholds_mm": list(thresholds),
        "num_tasks": len(tasks),
        "num_evaluated": len(ok),
        "num_errors": sum(1 for r in records if r.get("error")),
        "means": aggregate_metrics(records, keys),
        "mean_cg_chamfer_distance": float(np.mean(cg_chamfer)) if cg_chamfer else None,
        "mean_enh_chamfer_distance": float(np.mean(enh_chamfer)) if enh_chamfer else None,
        "mean_improvement_cg_minus_enh": float(np.mean(deltas)) if deltas else None,
        "note": "chamfer_distance = (accuracy + completeness) / 2, aligned test vs HE",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    out_json = args.out_json or os.path.join(
        args.test_root or GC2026_ROOT,
        f"evaluation_gc_baseline_{args.test_mode}.json",
    )
    out_csv = args.out_csv or out_json.replace(".json", ".csv")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    write_csv(out_csv, records, thresholds)

    print(json.dumps(summary, indent=2))
    print(f"Written: {out_json}")
    print(f"Written: {out_csv}")


if __name__ == "__main__":
    main()
