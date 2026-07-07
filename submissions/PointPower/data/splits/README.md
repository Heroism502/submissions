# Train / Val / SuperPC Skip — Auto-generated from Folder Names

This package does **not** hard-code sequence names.  
`bash data/generate_pair_lists.sh` scans directory names and writes:

- `data/splits/split.json` (auto-generated)
- `data/processed/runtime_gate.json` (`frame_fill_gate_skip_sequences`)

## Method A — Raw layout with train/val subfolders (recommended)

```
data/raw/UVG-CWI-DQPC/train/<Sequence>/consumer-grade_capture_system/CG/15fps/
data/raw/UVG-CWI-DQPC/val/<Sequence>/consumer-grade_capture_system/CG/15fps/
```

The script reads folder names under `train/`, `val/` (or `test/`) as the sequence list.

## Method B — Flat raw + split marker directories

If CG data is in a flat layout:

```
data/raw/UVG-CWI-DQPC/<Sequence>/consumer-grade_capture_system/CG/15fps/
```

Define the split with empty marker directories (directory name = sequence name):

```
data/splits/val/<Sequence>/          # one empty folder per validation sequence
data/splits/train/<Sequence>/        # optional; if omitted, train = all CG − val
```

## SuperPC full-sequence skip (optional)

Create empty folders under `data/splits/no_superpc_fill/<Sequence>/` for sequences that must skip SuperPC fill.  
Names are written to `data/processed/runtime_gate.json` → `frame_fill_gate_skip_sequences`.

## Generate lists

```bash
bash data/generate_pair_lists.sh
```

## Manual override (optional)

To override auto-discovery, provide `data/splits/split.json` (see `split.json.example`).
