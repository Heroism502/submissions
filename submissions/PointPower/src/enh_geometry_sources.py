"""Registered geometry caches for Phase-2 refine (independent of live PD-LTS)."""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

GC2026_ROOT = os.environ.get("GC2026_ROOT", "").strip()
if not GC2026_ROOT:
    _script = os.path.dirname(os.path.abspath(__file__))
    if _script not in sys.path:
        sys.path.insert(0, _script)
    from gc2026_paths import resolve_gc2026_root  # noqa: WPS433

    GC2026_ROOT = resolve_gc2026_root(_script)
else:
    GC2026_ROOT = os.path.abspath(GC2026_ROOT)

# name -> relative path under GC2026_ROOT; must use Seq/ENH_*.ply layout
DEFAULT_SOURCES: Dict[str, str] = {
    "superpc_filter_cg": "output/val_grid_official565/kitti360_com_filter_cg_v0_vx0",
    "superpc_blend_vx1": "output/val_grid_official565/kitti360_com_blend_cg_v0_vx1.0",
    "superpc_submission": "output/submission_candidate",
    "pdlts_light": "output/pdlts_val565/light",
    "pdlts_heavy": "output/pdlts_val565/heavy",
    "pdlts_finetune_uvg": "output/pdlts_finetune_uvg/val565/light",
    "superpc_uvg_pipeline": "output/superpc_uvg_pipeline/val565",
}


def resolve_source(name: str, root: str = GC2026_ROOT) -> Optional[str]:
    rel = DEFAULT_SOURCES.get(name)
    if not rel:
        return None
    path = os.path.join(root, rel)
    return path if os.path.isdir(path) else None


def count_cached_frames(source_dir: str) -> int:
    if not source_dir or not os.path.isdir(source_dir):
        return 0
    n = 0
    for _root, _dirs, files in os.walk(source_dir):
        n += sum(1 for f in files if f.endswith(".ply"))
    return n


def source_status(root: str = GC2026_ROOT) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, rel in DEFAULT_SOURCES.items():
        path = os.path.join(root, rel)
        n = count_cached_frames(path)
        out[name] = {
            "path": path,
            "exists": os.path.isdir(path),
            "num_ply": n,
            "ready_for_val565": n >= 565,
        }
    return out


def pick_geometry_dir(cfg, args_geometry_dir: str = "", root: str = GC2026_ROOT) -> Tuple[str, str]:
    """Return (absolute_geometry_dir, source_label)."""
    if args_geometry_dir:
        return args_geometry_dir, "cli"
    if cfg.geometry_dir:
        return cfg.geometry_dir, "config"
    src = (cfg.extra or {}).get("geometry_source")
    if src:
        resolved = resolve_source(src, root)
        if resolved:
            return resolved, src
    return "", ""


def pick_secondary_geometry_dir(cfg, args_geometry_dir: str = "", root: str = GC2026_ROOT) -> Tuple[str, str]:
    """Return secondary cache dir for hybrid presets (SuperPC + PD-LTS)."""
    if args_geometry_dir:
        return args_geometry_dir, "cli_secondary"
    sec = (cfg.extra or {}).get("geometry_secondary")
    if sec:
        resolved = resolve_source(sec, root)
        if resolved:
            return resolved, sec
    sec_dir = (cfg.extra or {}).get("geometry_secondary_dir")
    if sec_dir:
        path = sec_dir if os.path.isabs(sec_dir) else os.path.join(root, sec_dir)
        if os.path.isdir(path):
            return path, "secondary_dir"
    return "", ""
