#!/usr/bin/env python3
"""Evaluate UVG enhanced point clouds vs HE (official GC baseline metric).

Uses the same definitions as UVG-CWI Metric / evaluate_gc_baseline_metrics.py:
  - Per-sequence alignment matrix (CG→HE) applied to the test cloud only
  - chamfer_distance = 0.5 * (accuracy + completeness)

Requires: bash src/download_metric.sh  →  code/Metric/matrices/<Seq>.txt

Legacy flags --device / --n-samples are accepted for CLI compatibility but ignored.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from gc2026_paths import resolve_gc2026_root  # noqa: E402

GC2026_ROOT = resolve_gc2026_root(SCRIPT_DIR)


def enh_path_from_cg(cg_path: str, enhanced_root: str) -> str:
    """Map official CG path to ENH output path under enhanced_root/<Seq>/."""
    seq = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cg_path)))))
    fname = os.path.basename(cg_path).replace("_CG_", "_ENH_", 1)
    return os.path.join(enhanced_root, seq, fname)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ENH/CG vs HE with official aligned GC baseline Chamfer (mm)"
    )
    p.add_argument(
        "--pairs-file",
        default=os.path.join(GC2026_ROOT, "data", "processed", "val_pairs_official_cgv2.txt"),
    )
    p.add_argument(
        "--enhanced-root",
        default=None,
        help="Root with per-sequence ENH PLY folders (required unless --test-root set)",
    )
    p.add_argument(
        "--test-root",
        default=None,
        help="Alias for --enhanced-root (same as evaluate_gc_baseline_metrics.py)",
    )
    p.add_argument("--max-samples", type=int, default=0, help="Alias for --max-frames")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all pairs in file")
    p.add_argument("--max-load-points", type=int, default=0, help="0 = load all PLY points")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--thresholds", default="10,20,30,50")
    p.add_argument(
        "--also-cg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also score aligned CG vs HE (default: true)",
    )
    # Backward compatibility — ignored (official metric uses aligned full-cloud KDTree)
    p.add_argument("--n-samples", type=int, default=20000, help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_root = args.test_root or args.enhanced_root
    if not test_root:
        raise SystemExit("Provide --enhanced-root (or --test-root)")

    max_frames = args.max_frames or args.max_samples
    if args.n_samples != 20000 or args.device != "cuda":
        warnings.warn(
            "[evaluate_uvg] --n-samples/--device are ignored; "
            "using official aligned GC metric (full points, KDTree).",
            stacklevel=1,
        )

    from evaluate_gc_baseline_metrics import main as gc_main  # noqa: WPS433

    argv = [
        "evaluate_gc_baseline_metrics.py",
        "--pairs-file",
        args.pairs_file,
        "--test-root",
        test_root,
        "--test-mode",
        "enh",
        "--thresholds",
        args.thresholds,
        "--max-load-points",
        str(args.max_load_points),
        "--max-frames",
        str(max_frames),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
    ]
    if args.also_cg:
        argv.append("--also-cg")
    if args.out_json:
        argv.extend(["--out-json", args.out_json])
    if args.out_csv:
        argv.extend(["--out-csv", args.out_csv])

    old_argv = sys.argv
    try:
        sys.argv = argv
        gc_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
