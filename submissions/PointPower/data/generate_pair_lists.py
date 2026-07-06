#!/usr/bin/env python3
"""Discover train/val/skip sequences from folder names; write split + runtime gate config.

Discovery (first match wins):
  1. data/raw/UVG-CWI-DQPC/train/<Seq>/ and .../val/ or .../test/
  2. data/splits/train/<Seq>/ and data/splits/val/<Seq>/  (marker dirs, flat raw layout)
  3. data/splits/no_superpc_fill/<Seq>/  → frame_fill_gate_skip_sequences
  4. data/splits/split.json  (manual override only)
  5. data/splits/train_sequences.txt + val_sequences.txt
  6. TRAIN_SEQS / VAL_SEQS environment variables

Outputs (under GC2026_ROOT, auto-generated):
  data/splits/split.json
  data/processed/split_meta_official_cgv2.json
  data/processed/runtime_gate.json   ← merges skip list into submission preset
  data/processed/*.txt pair lists
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class SplitSpec:
    layout: str  # "flat" | "split_subdirs"
    train: List[str] = field(default_factory=list)
    val: List[str] = field(default_factory=list)
    all_sequences: List[str] = field(default_factory=list)
    no_superpc_fill: List[str] = field(default_factory=list)
    source: str = ""


def read_name_list(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return [ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.split("#", 1)[0].strip()]


def marker_dir_names(base: Path) -> List[str]:
    if not base.is_dir():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("."))


def cg_dir(root: Path, layout: str, split: str, seq: str) -> Path:
    if layout == "split_subdirs":
        return root / split / seq / "consumer-grade_capture_system" / "CG" / "15fps"
    return root / seq / "consumer-grade_capture_system" / "CG" / "15fps"


def he_dir(root: Path, layout: str, split: str, seq: str) -> Path:
    if layout == "split_subdirs":
        return root / split / seq / "high-end_capture_system" / "HE" / "15fps"
    return root / seq / "high-end_capture_system" / "HE" / "15fps"


def has_cg(root: Path, layout: str, split: str, seq: str) -> bool:
    d = cg_dir(root, layout, split, seq)
    return d.is_dir() and any(d.glob("*.ply"))


def discover_flat_sequences(raw_root: Path) -> List[str]:
    if not raw_root.is_dir():
        return []
    return sorted(
        d.name
        for d in raw_root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in ("train", "val", "test")
        and has_cg(raw_root, "flat", "", d.name)
    )


def discover_subdir_sequences(raw_root: Path, sub: str) -> List[str]:
    base = raw_root / sub
    if not base.is_dir():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and has_cg(raw_root, "split_subdirs", sub, d.name))


def discover_no_superpc_fill(splits_dir: Path, all_seqs: List[str]) -> Tuple[List[str], str]:
    marker = splits_dir / "no_superpc_fill"
    if marker.is_dir():
        names = [n for n in marker_dir_names(marker) if n in all_seqs]
        if names:
            return sorted(names), str(marker)
    txt = splits_dir / "no_superpc_fill_sequences.txt"
    names = [n for n in read_name_list(txt) if n in all_seqs]
    if names:
        return sorted(names), str(txt)
    return [], ""


def resolve_split(raw_root: Path, splits_dir: Path) -> SplitSpec:
    # 1) raw/train + raw/val|test subdirs
    train_sub = discover_subdir_sequences(raw_root, "train")
    val_sub = discover_subdir_sequences(raw_root, "val")
    test_sub = discover_subdir_sequences(raw_root, "test")
    if train_sub or val_sub or test_sub:
        val_merged = sorted(set(val_sub) | set(test_sub))
        all_seqs = sorted(set(train_sub) | set(val_merged))
        return SplitSpec(
            layout="split_subdirs",
            train=train_sub,
            val=val_merged,
            all_sequences=all_seqs,
            source=f"{raw_root}/{{train,val,test}}/",
        )

    flat_all = discover_flat_sequences(raw_root)
    if not flat_all:
        raise SystemExit(f"No CG sequences under {raw_root}")

    # 2) marker dirs data/splits/train|val/<Seq>/
    train_markers = marker_dir_names(splits_dir / "train")
    val_markers = marker_dir_names(splits_dir / "val") or marker_dir_names(splits_dir / "test")
    if train_markers or val_markers:
        train = sorted(set(train_markers) & set(flat_all)) if train_markers else sorted(set(flat_all) - set(val_markers))
        val = sorted(set(val_markers) & set(flat_all))
        if not train and val:
            train = sorted(set(flat_all) - set(val))
        return SplitSpec(
            layout="flat",
            train=train,
            val=val,
            all_sequences=flat_all,
            source=f"{splits_dir}/{{train,val}}/ markers",
        )

    # 3) split.json override
    sj = splits_dir / "split.json"
    if sj.is_file() and os.environ.get("IGNORE_SPLIT_JSON", "") != "1":
        data = json.loads(sj.read_text(encoding="utf-8"))
        train = [s for s in (data.get("train") or []) if s in flat_all]
        val = [s for s in (data.get("val") or data.get("test") or []) if s in flat_all]
        if train or val:
            return SplitSpec(
                layout="flat",
                train=train or sorted(set(flat_all) - set(val)),
                val=val,
                all_sequences=flat_all,
                source=f"{sj} (override)",
            )

    train_txt = read_name_list(splits_dir / "train_sequences.txt")
    val_txt = read_name_list(splits_dir / "val_sequences.txt") or read_name_list(splits_dir / "test_sequences.txt")
    if train_txt or val_txt:
        train = [s for s in train_txt if s in flat_all] if train_txt else sorted(set(flat_all) - set(val_txt))
        val = [s for s in val_txt if s in flat_all]
        return SplitSpec(
            layout="flat",
            train=train,
            val=val,
            all_sequences=flat_all,
            source=str(splits_dir),
        )

    env_train = [s for s in os.environ.get("TRAIN_SEQS", "").split() if s in flat_all]
    env_val = [s for s in (os.environ.get("VAL_SEQS", "") or os.environ.get("TEST_SEQS", "")).split() if s in flat_all]
    if env_train or env_val:
        return SplitSpec(
            layout="flat",
            train=env_train or sorted(set(flat_all) - set(env_val)),
            val=env_val,
            all_sequences=flat_all,
            source="environment",
        )

    raise SystemExit(
        "Found CG data but cannot infer train/val split from folder names.\n"
        "Provide ONE of (relative to GC2026_ROOT):\n"
        "  data/raw/UVG-CWI-DQPC/train/<Seq>/ and .../val/<Seq>/\n"
        "  data/splits/val/<Seq>/ marker dirs (flat raw layout)\n"
        "  data/splits/split.json override\n"
        "See data/splits/README.md"
    )


def split_for_seq(spec: SplitSpec, seq: str) -> str:
    if seq in spec.train:
        return "train"
    if seq in spec.val:
        return "val"
    return ""


def iter_cg_files(root: Path, layout: str, split: str, seq: str) -> List[str]:
    d = cg_dir(root, layout, split, seq)
    return sorted(str(p) for p in d.glob("*.ply")) if d.is_dir() else []


def write_cg_list(path: Path, root: Path, spec: SplitSpec, seqs: List[str]) -> int:
    lines: List[str] = []
    for seq in seqs:
        if spec.layout == "split_subdirs":
            sp = split_for_seq(spec, seq)
            if sp:
                lines.extend(iter_cg_files(root, spec.layout, sp, seq))
        else:
            lines.extend(iter_cg_files(root, "flat", "", seq))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_pairs_list(path: Path, root: Path, spec: SplitSpec, seqs: List[str]) -> Tuple[int, int]:
    lines: List[str] = []
    missing = 0
    for seq in seqs:
        if spec.layout == "split_subdirs":
            sp = split_for_seq(spec, seq)
            if not sp:
                continue
            cg_files = iter_cg_files(root, spec.layout, sp, seq)
            he_base = he_dir(root, spec.layout, sp, seq)
        else:
            cg_files = iter_cg_files(root, "flat", "", seq)
            he_base = he_dir(root, "flat", "", seq)
        for cg_ply in cg_files:
            he_ply = he_base / Path(cg_ply).name.replace("_CG_", "_HE_", 1)
            if he_ply.is_file():
                lines.append(f"{cg_ply}\t{he_ply}")
            else:
                missing += 1
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), missing


def write_split_json(splits_dir: Path, spec: SplitSpec) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "_generated": True,
        "train": spec.train,
        "val": spec.val,
        "no_superpc_fill": spec.no_superpc_fill,
        "source": spec.source,
    }
    (splits_dir / "split.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_runtime_gate(gc_root: Path, submission_root: Path, skip_seqs: List[str]) -> Path:
    base_path = submission_root / "config/gate_decision.json"
    gate = json.loads(base_path.read_text(encoding="utf-8"))
    for key in ("production_config", "best_config"):
        block = gate.get(key) or {}
        extra = dict(block.get("extra") or {})
        extra["frame_fill_gate_skip_sequences"] = list(skip_seqs)
        block["extra"] = extra
        gate[key] = block
    out = gc_root / "data/processed/runtime_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    gc_root = Path(os.environ.get("GC2026_ROOT", "./workspace")).resolve()
    raw = Path(os.environ.get("UVG_RAW_DIR", gc_root / "data/raw/UVG-CWI-DQPC"))
    splits_dir = Path(os.environ.get("SPLITS_DIR", gc_root / "data/splits"))
    out = Path(os.environ.get("PAIR_LIST_OUT", gc_root / "data/processed"))
    submission_root = Path(os.environ.get("SUBMISSION_ROOT", Path(__file__).resolve().parent.parent))
    out.mkdir(parents=True, exist_ok=True)

    spec = resolve_split(raw, splits_dir)
    skip_seqs, skip_src = discover_no_superpc_fill(splits_dir, spec.all_sequences)
    spec.no_superpc_fill = skip_seqs

    if spec.source != f"{splits_dir / 'split.json'} (override)":
        write_split_json(splits_dir, spec)

    paths = {
        "all_cg": out / "all_cg_only_cgv2.txt",
        "all_pairs": out / "all_pairs_cgv2.txt",
        "train_pairs": out / "train_pairs_official_cgv2.txt",
        "val_pairs": out / "val_pairs_official_cgv2.txt",
    }
    n_all_cg = write_cg_list(paths["all_cg"], raw, spec, spec.all_sequences)
    n_all_pairs, m_all = write_pairs_list(paths["all_pairs"], raw, spec, spec.all_sequences)
    n_train, m_train = write_pairs_list(paths["train_pairs"], raw, spec, spec.train)
    n_val, m_val = write_pairs_list(paths["val_pairs"], raw, spec, spec.val)

    gate_path = write_runtime_gate(gc_root, submission_root, skip_seqs)

    meta = {
        "cg_version": "v2",
        "split_source": spec.source,
        "skip_superpc_source": skip_src or "(none)",
        "layout": spec.layout,
        "all_sequences": spec.all_sequences,
        "train_sequences": spec.train,
        "val_sequences": spec.val,
        "no_superpc_fill_sequences": skip_seqs,
        "num_all_cg": n_all_cg,
        "num_train_pairs": n_train,
        "num_val_pairs": n_val,
        "runtime_gate_file": "data/processed/runtime_gate.json",
    }
    (out / "split_meta_official_cgv2.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"[generate_pair_lists] GC2026_ROOT={gc_root}")
    print(f"[generate_pair_lists] split_source={spec.source}")
    print(f"[generate_pair_lists] layout={spec.layout} all={len(spec.all_sequences)} train={len(spec.train)} val={len(spec.val)}")
    print(f"[generate_pair_lists] train: {' '.join(spec.train) or '(none)'}")
    print(f"[generate_pair_lists] val:   {' '.join(spec.val) or '(none)'}")
    print(f"[generate_pair_lists] no_superpc_fill: {' '.join(skip_seqs) or '(none)'}")
    print(f"[generate_pair_lists] wrote data/splits/split.json")
    print(f"[generate_pair_lists] wrote data/processed/runtime_gate.json")
    if m_all + m_train + m_val:
        print(f"[generate_pair_lists] WARN: {m_all + m_train + m_val} CG without HE", file=sys.stderr)
    print(f"[generate_pair_lists] {n_all_cg} all CG | {n_train} train pairs | {n_val} val pairs")
    print("[generate_pair_lists] DONE")


if __name__ == "__main__":
    main()
