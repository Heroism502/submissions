"""Configuration and presets for multi-stage Enh refinement pipeline."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RefineConfig:
    """One reproducible Enh refinement recipe."""

    name: str = "cg_passthrough"
    geometry: str = "passthrough_cg"
    geometry_dir: str = ""
    pre_sor: bool = False
    pre_sor_nb: int = 20
    pre_sor_std: float = 2.5
    snap_mm: float = 0.0
    fill_mm: float = 0.0
    post_sor: bool = False
    post_sor_nb: int = 20
    post_sor_std: float = 2.5
    blend_voxel_mm: float = 0.0
    adaptive_route: bool = False
    temporal_window: int = 0
    pdlts_model: str = "light"
    pdlts_cluster_size: int = 50000
    geometry_fallback: str = "filter_cg"
    bidirectional_snap: bool = False
    cg_pull_mm: float = 1.0
    fill_mode: str = "fixed"  # fixed | density_adaptive
    fill_density_k: int = 6
    fill_density_scale_max: float = 2.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def tag(self) -> str:
        if self.name:
            return re.sub(r"[^a-zA-Z0-9._-]+", "_", self.name)
        parts = [self.geometry]
        if self.pre_sor:
            parts.append(f"pre{self.pre_sor_std:g}")
        if self.snap_mm > 0:
            parts.append(f"snap{self.snap_mm:g}")
        if self.fill_mm > 0:
            parts.append(f"fill{self.fill_mm:g}")
        if self.post_sor:
            parts.append(f"post{self.post_sor_std:g}")
        if self.blend_voxel_mm > 0:
            parts.append(f"vx{self.blend_voxel_mm:g}")
        if self.adaptive_route:
            parts.append("adapt")
        if self.bidirectional_snap:
            parts.append("bidir")
        if self.fill_mode == "density_adaptive":
            parts.append("dfill")
        return "_".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RefineConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs)
        if extra:
            cfg.extra.update(extra)
        return cfg

    @classmethod
    def from_json_file(cls, path: str) -> "RefineConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


ROLLBACK_CONFIG = RefineConfig(
    name="cg_passthrough",
    geometry="passthrough_cg",
)

FILTER_CG_CONFIG = RefineConfig(
    name="filter_cg",
    geometry="filter_cg",
)


def _pdlts(name: str, model: str = "light", **kwargs) -> RefineConfig:
    base = dict(
        geometry=f"pdlts_{model}",
        pdlts_model=model,
    )
    base.update(kwargs)
    return RefineConfig(name=name, **base)


def _hybrid_extra(
    secondary: str = "superpc_submission",
    fill_role: str = "union_voxel",
    voxel_mm: float = 0.5,
) -> dict:
    return {
        "geometry_source": "pdlts_light",
        "geometry_secondary": secondary,
        "hybrid_fill_role": fill_role,
        "hybrid_voxel_mm": voxel_mm,
    }


def _holefill_extra(**kwargs) -> dict:
    return {
        **_hybrid_extra(fill_role="secondary_only"),
        "hybrid_fill_anchor": "primary",
        "hybrid_hole_mask": "cg_holes",
        **kwargs,
    }


# ft density refine applied to primary before optional SuperPC (architecture v2).
PRIMARY_DENSITY_REFINE = {
    "snap_mm": 1.0,
    "fill_mm": 0.6,
    "fill_mode": "density_adaptive",
    "fill_density_k": 6,
    "fill_density_scale_max": 2.0,
}


def _hybrid(name: str, **kwargs) -> RefineConfig:
    extra = _hybrid_extra()
    snap = kwargs.pop("snap_mm", 1.0)
    fill = kwargs.pop("fill_mm", 0.6)
    extra.update(kwargs.pop("extra", {}))
    return RefineConfig(
        name=name,
        geometry="hybrid_pdlts_superpc",
        snap_mm=snap,
        fill_mm=fill,
        extra=extra,
        **kwargs,
    )


def _from_src(name: str, source: str, **kwargs) -> RefineConfig:
    return RefineConfig(
        name=name,
        geometry="from_dir",
        extra={"geometry_source": source},
        **kwargs,
    )


# Phase 2A: no external geometry (always runnable)
PHASE2A_PRESETS: List[str] = ["cg_passthrough", "filter_cg"]

# Phase 2B: use existing full val565 caches (SuperPC etc.)
PHASE2B_PRESETS: List[str] = [
    "superpc_filter_snap1.0",
    "superpc_filter_snap1_fill0.6",
    "superpc_filter_post25",
]

# Phase 2C: needs PD-LTS cache (partial OK with geometry_fallback)
PHASE2C_PRESETS: List[str] = [
    "pdlts_light_snap1.0",
    "pdlts_light_snap1_fill0.6",
    "pdlts_light_snap1_fill0.6_post25",
    "pdlts_light_snap1_adapt",
]

# Phase 2D: snap/fill fine grid on PD-LTS light cache (CPU post-process only)
SNAP_FILL_SNAP_MM = (0.5, 1.0, 1.2, 1.5)
SNAP_FILL_FILL_MM = (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
PHASE2D_EXTRA_PRESETS: List[str] = [
    "pdlts_light_snap1_fill0.6_post25",
    "pdlts_light_snap1_adapt",
]


def snap_fill_preset_name(snap_mm: float, fill_mm: float) -> str:
    return f"pdlts_light_snap{snap_mm:g}_fill{fill_mm:g}"


def make_snap_fill_preset(snap_mm: float, fill_mm: float) -> RefineConfig:
    return _pdlts(snap_fill_preset_name(snap_mm, fill_mm), "light", snap_mm=snap_mm, fill_mm=fill_mm)


def build_snap_fill_grid_presets(
    base_presets: List[RefineConfig],
    snaps: tuple[float, ...] = SNAP_FILL_SNAP_MM,
    fills: tuple[float, ...] = SNAP_FILL_FILL_MM,
) -> List[RefineConfig]:
    seen = {p.name for p in base_presets}
    out: List[RefineConfig] = []
    for snap in snaps:
        for fill in fills:
            cfg = make_snap_fill_preset(snap, fill)
            if cfg.name in seen:
                continue
            out.append(cfg)
            seen.add(cfg.name)
    return out


GRID_PRESETS: List[RefineConfig] = [
    ROLLBACK_CONFIG,
    FILTER_CG_CONFIG,
    _from_src("superpc_filter_snap1.0", "superpc_filter_cg", snap_mm=1.0),
    _from_src("superpc_filter_snap1_fill0.6", "superpc_filter_cg", snap_mm=1.0, fill_mm=0.6),
    _from_src("superpc_filter_post25", "superpc_filter_cg", post_sor=True, post_sor_std=2.5),
    _pdlts("pdlts_light", "light"),
    _pdlts("pdlts_heavy", "heavy"),
    _pdlts("pdlts_light_snap0.5", "light", snap_mm=0.5),
    _pdlts("pdlts_light_snap1.0", "light", snap_mm=1.0),
    _pdlts("pdlts_light_snap1.5", "light", snap_mm=1.5),
    _pdlts("pdlts_light_snap1_fill0.6", "light", snap_mm=1.0, fill_mm=0.6),
    _pdlts("pdlts_light_snap1_fill1.0", "light", snap_mm=1.0, fill_mm=1.0),
    _pdlts("pdlts_light_pre25_snap1", "light", pre_sor=True, pre_sor_std=2.5, snap_mm=1.0),
    _pdlts("pdlts_light_snap1_post25", "light", snap_mm=1.0, post_sor=True, post_sor_std=2.5),
    _pdlts("pdlts_light_snap1_fill0.6_post25", "light", snap_mm=1.0, fill_mm=0.6, post_sor=True, post_sor_std=2.5),
    _pdlts("pdlts_heavy_snap1.0", "heavy", snap_mm=1.0),
    _pdlts("pdlts_light_snap1_adapt", "light", snap_mm=1.0, fill_mm=0.6, adaptive_route=True),
    _pdlts(
        "pdlts_light_snap1_fill0.6_density",
        "light",
        snap_mm=1.0,
        fill_mm=0.6,
        fill_mode="density_adaptive",
    ),
    _pdlts(
        "pdlts_light_snap1_fill0.6_bidir",
        "light",
        snap_mm=1.0,
        fill_mm=0.6,
        bidirectional_snap=True,
        cg_pull_mm=1.0,
    ),
    _pdlts(
        "pdlts_light_snap1_fill0.6_combined",
        "light",
        snap_mm=1.0,
        fill_mm=0.6,
        fill_mode="density_adaptive",
        bidirectional_snap=True,
        cg_pull_mm=1.0,
    ),
    _hybrid(
        "hybrid_pdlts_superpc_snap1_fill0.6_density",
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=2.0,
    ),
    _hybrid(
        "hybrid_pdlts_superpc_snap1_fill0.6_superfill",
        fill_mode="fixed",
        extra=_hybrid_extra(fill_role="secondary_only"),
    ),
    _hybrid(
        "region_hybrid_pdlts_superpc_snap1_fill0.6_density",
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=2.0,
        extra={
            **_hybrid_extra(fill_role="region_mask"),
            "region_r_in_mm": 25.0,
            "region_r_out_mm": 45.0,
        },
    ),
    _hybrid(
        "temporal_region_hybrid_pdlts_superpc_snap1_fill0.6_density",
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=2.0,
        temporal_window=5,
        extra={
            **_hybrid_extra(fill_role="temporal_region_mask"),
            "temporal_match_mm": 15.0,
            "temporal_tau_interior": 0.6,
            "temporal_tau_exterior": 0.2,
            "temporal_cg_link_mm": 25.0,
        },
    ),
    _hybrid(
        "temporal_attn_hybrid_pdlts_superpc_snap1_fill0.6_density",
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=2.0,
        temporal_window=15,
        extra={
            **_hybrid_extra(fill_role="temporal_attn_region_mask"),
            "temporal_match_mm": 15.0,
            "temporal_tau_interior": 0.6,
            "temporal_tau_exterior": 0.2,
            "temporal_cg_link_mm": 25.0,
            "temporal_attn_tau_time": 8.0,
            "temporal_attn_tau_dist": 12.0,
            "temporal_disp_agree_mm": 8.0,
            "temporal_cg_weight": 0.35,
            "temporal_enh_weight": 0.65,
        },
    ),
    # Primary-heavy hybrid: less SuperPC fill, tighter CG-interior mask (preserve ft geometry).
    _hybrid(
        "region_hybrid_pdlts_superpc_snap1_fill0.25_density_rin15",
        fill_mm=0.25,
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=1.5,
        extra={
            **_hybrid_extra(fill_role="region_mask"),
            "region_r_in_mm": 15.0,
            "region_r_out_mm": 30.0,
        },
    ),
    _hybrid(
        "region_hybrid_pdlts_superpc_snap1_fill0.35_density_rin18",
        fill_mm=0.35,
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=1.8,
        extra={
            **_hybrid_extra(fill_role="region_mask"),
            "region_r_in_mm": 18.0,
            "region_r_out_mm": 35.0,
        },
    ),
    _hybrid(
        "region_hybrid_pdlts_superpc_snap1_fill0.2_density_rin12",
        fill_mm=0.2,
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=1.3,
        extra={
            **_hybrid_extra(fill_role="region_mask"),
            "region_r_in_mm": 12.0,
            "region_r_out_mm": 25.0,
        },
    ),
    # CG-hole fill on primary (ft) surface; secondary SuperPC only where CG sparse.
    _hybrid(
        "holefill_secondary_cg_hybrid_pdlts_superpc_snap1_fill0.6_density",
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=2.0,
        extra=_holefill_extra(),
    ),
    # Fill first, snap, then post-SOR denoise on merged cloud.
    _hybrid(
        "holefill_first_secondary_cg_hybrid_pdlts_superpc_fill0.6_post25_density",
        snap_mm=1.0,
        fill_mm=0.6,
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=2.0,
        post_sor=True,
        post_sor_std=2.5,
        extra=_holefill_extra(fill_before_snap=True),
    ),
    # Lower SuperPC weight: smaller fill, cap added pts, skip post-SOR when fill is tiny.
    _hybrid(
        "holefill_lite_fill0.25_max3pct_nopost_snap0",
        snap_mm=0.0,
        fill_mm=0.25,
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=1.3,
        extra=_holefill_extra(
            fill_before_snap=True,
            hybrid_max_fill_ratio=0.03,
        ),
    ),
    _hybrid(
        "holefill_lite_fill0.25_max10pct_adaptive_post25",
        snap_mm=0.0,
        fill_mm=0.25,
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=1.3,
        post_sor=True,
        post_sor_std=2.5,
        extra=_holefill_extra(
            fill_before_snap=True,
            hybrid_max_fill_ratio=0.10,
            adaptive_post_sor=True,
            adaptive_post_sor_min_add_ratio=0.02,
        ),
    ),
    _hybrid(
        "holefill_adaptive_frame_gate",
        snap_mm=0.0,
        fill_mm=0.25,
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=1.3,
        post_sor=True,
        post_sor_std=2.5,
        extra=_holefill_extra(
            fill_before_snap=True,
            hybrid_max_fill_ratio=0.10,
            adaptive_post_sor=True,
            adaptive_post_sor_min_add_ratio=0.02,
            frame_fill_gate=True,
            frame_fill_gate_probe_fill_mm=0.25,
            frame_fill_gate_probe_scale_max=1.3,
            frame_fill_gate_probe_max_ratio=0.10,
            frame_fill_gate_probe_k=6,
            frame_fill_gate_skip_add_ratio=0.022,
            frame_fill_gate_full_add_ratio=0.040,
            frame_fill_gate_k=6,
            frame_fill_gate_tiers={
                "skip": {"fill_mm": 0.0, "post_sor": False},
                "lite": {
                    "fill_mm": 0.25,
                    "fill_density_scale_max": 1.3,
                    "hybrid_max_fill_ratio": 0.10,
                    "post_sor": False,
                    "adaptive_post_sor": True,
                    "adaptive_post_sor_min_add_ratio": 0.02,
                },
                "full": {
                    "fill_mm": 0.6,
                    "fill_density_scale_max": 2.0,
                    "hybrid_max_fill_ratio": 0.15,
                    "post_sor": True,
                    "post_sor_std": 2.5,
                    "adaptive_post_sor": False,
                },
            },
        ),
    ),
    _hybrid(
        "holefill_adaptive_frame_gate_v2",
        snap_mm=0.0,
        fill_mm=0.25,
        fill_mode="density_adaptive",
        fill_density_k=6,
        fill_density_scale_max=1.3,
        post_sor=True,
        post_sor_std=2.5,
        extra=_holefill_extra(
            primary_density_refine=dict(PRIMARY_DENSITY_REFINE),
            fill_before_snap=True,
            hybrid_max_fill_ratio=0.10,
            adaptive_post_sor=True,
            adaptive_post_sor_min_add_ratio=0.02,
            frame_fill_gate=True,
            frame_fill_gate_skip_sequences=[],
            frame_fill_gate_probe_fill_mm=0.25,
            frame_fill_gate_probe_scale_max=1.3,
            frame_fill_gate_probe_max_ratio=0.10,
            frame_fill_gate_probe_k=6,
            frame_fill_gate_skip_add_ratio=0.022,
            frame_fill_gate_full_add_ratio=0.040,
            frame_fill_gate_k=6,
            frame_fill_gate_tiers={
                "skip": {"fill_mm": 0.0, "post_sor": False},
                "lite": {
                    "fill_mm": 0.25,
                    "fill_density_scale_max": 1.3,
                    "hybrid_max_fill_ratio": 0.10,
                    "post_sor": False,
                    "adaptive_post_sor": True,
                    "adaptive_post_sor_min_add_ratio": 0.02,
                },
                "full": {
                    "fill_mm": 0.6,
                    "fill_density_scale_max": 2.0,
                    "hybrid_max_fill_ratio": 0.15,
                    "post_sor": True,
                    "post_sor_std": 2.5,
                    "adaptive_post_sor": False,
                },
            },
        ),
    ),
    RefineConfig(
        name="pdlts_light_snap1_fill0.6_density_temporal5",
        geometry="from_dir",
        snap_mm=1.0,
        fill_mm=0.6,
        fill_mode="density_adaptive",
        temporal_window=5,
        extra={"geometry_source": "pdlts_light", "post_temporal_smooth": True},
    ),
    RefineConfig(
        name="fp_migrated_pre25_density",
        geometry="from_dir",
        pre_sor=True,
        pre_sor_std=2.5,
        snap_mm=1.0,
        fill_mm=0.6,
        fill_mode="density_adaptive",
        adaptive_route=True,
        extra={"geometry_source": "pdlts_light", "fp_filter_before_fill": True},
    ),
    RefineConfig(
        name="from_dir_snap1_fill0.6",
        geometry="from_dir",
        snap_mm=1.0,
        fill_mm=0.6,
    ),
]

SNAP_FILL_GRID_PRESETS: List[RefineConfig] = build_snap_fill_grid_presets(GRID_PRESETS)
PHASE2D_PRESETS: List[str] = [p.name for p in SNAP_FILL_GRID_PRESETS] + PHASE2D_EXTRA_PRESETS

PRESET_BY_NAME = {p.name: p for p in GRID_PRESETS}
for _cfg in SNAP_FILL_GRID_PRESETS:
    PRESET_BY_NAME[_cfg.name] = _cfg
PRESET_BY_NAME[ROLLBACK_CONFIG.name] = ROLLBACK_CONFIG

ALL_PHASE2_PRESETS = PHASE2A_PRESETS + PHASE2B_PRESETS + PHASE2C_PRESETS + PHASE2D_PRESETS


def resolve_preset(name: str) -> RefineConfig:
    if name in PRESET_BY_NAME:
        return PRESET_BY_NAME[name]
    m = re.match(r"^pdlts_light_snap([\d.]+)_fill([\d.]+)$", name)
    if m:
        cfg = make_snap_fill_preset(float(m.group(1)), float(m.group(2)))
        PRESET_BY_NAME[name] = cfg
        return cfg
    raise KeyError(f"Unknown refine preset: {name}. Known: {sorted(PRESET_BY_NAME)}")
