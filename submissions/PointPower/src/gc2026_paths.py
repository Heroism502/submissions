"""Resolve GC2026 workspace root from submission package or project scripts/."""
from __future__ import annotations

import os


def resolve_gc2026_root(script_dir: str) -> str:
    env = os.environ.get("GC2026_ROOT", "").strip()
    if env:
        return os.path.abspath(env)
    sub_root = os.path.dirname(os.path.abspath(script_dir))
    # submission layout: .../submissions/TeamName/src -> workspace two levels up
    workspace = os.path.abspath(os.path.join(sub_root, "..", ".."))
    if os.path.isdir(os.path.join(workspace, "data")):
        code = os.path.join(workspace, "code")
        if os.path.isdir(os.path.join(code, "PD-LTS")) or os.path.isdir(
            os.path.join(code, "SuperPC")
        ):
            return workspace
    # project scripts/ layout
    if os.path.isdir(os.path.join(sub_root, "data")):
        return sub_root
    return workspace
