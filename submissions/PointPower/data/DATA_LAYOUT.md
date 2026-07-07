# Dataset Directory Layout (relative to `$GC2026_ROOT`)

## Consumer-grade / High-end data

```
data/raw/UVG-CWI-DQPC/<Sequence>/consumer-grade_capture_system/CG/15fps/*.ply
data/raw/UVG-CWI-DQPC/<Sequence>/high-end_capture_system/HE/15fps/*.ply
```

`<Sequence>` is the **folder name** as provided in the official dataset.

## Train / val split (from folder names)

See **`data/splits/README.md`**. Either layout works:

- `data/raw/UVG-CWI-DQPC/train/<Seq>/` + `.../val/<Seq>/`
- flat raw + empty marker dirs under `data/splits/val/<Seq>/`

After `bash data/generate_pair_lists.sh`:

| Output | Description |
|--------|-------------|
| `data/splits/split.json` | Detected train/val sequence names |
| `data/processed/runtime_gate.json` | Preset + `no_superpc_fill` sequences |
| `data/processed/*.txt` | CG / pair list files |

## SuperPC sequence-level skip (optional)

```
data/splits/no_superpc_fill/<Sequence>/    # empty directory; name = sequence
```

## Model checkpoints (bundled in submission)

| Purpose | Path |
|---------|------|
| PD-LTS inference | `models/DenoiseFlow-light-UVG-finetune.ckpt` |
| SuperPC | `models/kitti360_com.pth` |
| Runtime gating | `data/processed/runtime_gate.json` (auto-generated) |
