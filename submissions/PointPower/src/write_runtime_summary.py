#!/usr/bin/env python3
"""Write competition runtime summary (hardware + per-frame timing) for Full Pipeline."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime


def gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        names = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return ", ".join(names) if names else "none"
    except Exception:
        return "none"


def load_infer_records(infer_meta: str) -> list[dict]:
    if not os.path.isfile(infer_meta):
        return []
    with open(infer_meta, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", [])


def main() -> None:
    p = argparse.ArgumentParser(description="Build runtime log for GC2026 submission")
    p.add_argument("--out-dir", required=True, help="Enhanced output directory")
    p.add_argument("--infer-meta", default="", help="infer_meta.json path (default: out-dir/infer_meta.json)")
    p.add_argument("--stage1-meta", default="", help="Optional Stage1 timing JSON")
    p.add_argument("--team", default="GC2026 Team")
    args = p.parse_args()

    infer_path = args.infer_meta or os.path.join(args.out_dir, "infer_meta.json")
    records = load_infer_records(infer_path)

    total_sec = sum(float(r.get("seconds", 0)) for r in records)
    hardware = {
        "gpus": gpu_info(),
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
    }

    payload = {
        "team": args.team,
        "processing_track": "Full Pipeline",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "hardware": hardware,
        "stage2_superpc": {
            "infer_meta": os.path.abspath(infer_path),
            "num_frames": len(records),
            "total_seconds": round(total_sec, 3),
            "mean_seconds_per_frame": round(total_sec / len(records), 4) if records else 0.0,
        },
        "per_frame": records,
    }

    if args.stage1_meta and os.path.isfile(args.stage1_meta):
        with open(args.stage1_meta, encoding="utf-8") as f:
            payload["stage1"] = json.load(f)

    out_json = os.path.join(args.out_dir, "runtime_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_log = os.path.join(args.out_dir, "runtime.log")
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(f"team={args.team}\n")
        f.write(f"track=Full Pipeline\n")
        f.write(f"hardware_gpus={hardware['gpus']}\n")
        f.write(f"hardware_os={hardware['os']}\n")
        f.write(f"frames={len(records)}\n")
        f.write(f"total_seconds={payload['stage2_superpc']['total_seconds']}\n")
        f.write(f"mean_seconds_per_frame={payload['stage2_superpc']['mean_seconds_per_frame']}\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"runtime_summary.json -> {out_json}")
    print(f"runtime.log -> {out_log}")


if __name__ == "__main__":
    main()
