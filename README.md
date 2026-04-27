# UVG-CWI-DQPC: Dual-Quality Point Cloud Dataset — Grand Challenge Submissions

This repository hosts submissions for the **UVG-CWI-DQPC Grand Challenge**, which focuses on improving consumer-grade point clouds to match high-end capture quality.

**Dataset website:** [https://ultravideo.fi/UVG-CWI-DQPC/index.html](https://ultravideo.fi/UVG-CWI-DQPC/index.html)

## Challenge Overview

The [UVG-CWI-DQPC dataset](https://ultravideo.fi/UVG-CWI-DQPC/index.html) contains 12 volumetric video sequences captured simultaneously with both a **high-end multi-camera capture system** and **consumer-grade Intel RealSense depth cameras**. The goal of this challenge is to develop algorithms that enhance consumer-grade point clouds toward the quality of the high-end captures.

Participants may choose one of two processing tracks:

| Track | Input | Description |
|-------|-------|-------------|
| **Full Pipeline** | Intel RealSense `.bag` files (RGBD data) | Process raw depth sensor data end-to-end to produce enhanced PLY point clouds |
| **Enhancement Only** | Consumer-grade `.ply` files | Improve the already-extracted consumer-grade PLY point clouds |

### Evaluation

Submitted point clouds are evaluated by comparing each output `.ply` frame against the corresponding **High-Quality "ground truth"** `.ply` from the high-end capture system. The following metrics are computed:

- **Chamfer Distance** — Symmetric average nearest-neighbor distance between point clouds
- **Accuracy** — Average distance from each point in the output to its nearest neighbor in the ground truth
- **Completeness** — Average distance from each point in the ground truth to its nearest neighbor in the output
- **Runtime** — Total processing time for all frames (in seconds)

**NOTE** Metrics are subject to change; likely some additional metrics are added.

### Dataset Sequences

The dataset includes the following sequences:

| Sequence | Frames | Description |
|----------|--------|-------------|
| BlueSpeech | 169 | Person delivering a speech with hand gestures |
| BlueVolley | 171 | Person playing with a volleyball |
| BouncingBlue | 157 | Person bouncing on a gym ball |
| FitFluencer | 201 | Person stretching sideways |
| GoodVision | 168 | Person conducting an eye exam |
| Mannequin | 188 | Static mannequin wearing HMD and T-shirt |
| OrangeKettlebell | 170 | Person performing kettlebell swings |
| PinkNoir | 201 | Person posing for the camera |
| TicTacToe | 165 | Two persons playing Tic Tac Toe |
| TrumanShow | 171 | Person greeting cameras with gestures |
| VictoryHeart | 197 | Person making heart shape with hands |
| VirtualLife | 196 | Person in VR gameplay with HMD |

## How to Submit

### 1. Fork & Clone

Fork this repository and clone it locally.

### 2. Create Your Submission Directory

Create a directory under `submissions/` using your team or algorithm name:

```
submissions/
└── your_team_or_algorithm_name/
    ├── README.md           # Required: algorithm description (see below)
    ├── src/                # Required: source code to reproduce results
    │   └── ...
    └── requirements.txt    # Recommended: dependencies
```

> **Note:** Do not include output PLY files in your submission — they are too large for version control. The organizers will run your source code to generate and evaluate the results.

### 3. Write Your Submission README

Your `submissions/your_team_or_algorithm_name/README.md` **must** include:

- **Team Name**
- **Team Members** — Full names and affiliations
- **Algorithm Name** — Short identifier for your method
- **Algorithm Description** — Summary of your approach
- **Processing Track** — Either `Full Pipeline` (from `.bag` files) or `Enhancement Only` (from `.ply` files)
- **How to Run** — Clear instructions to reproduce your results from the source code in `src/`
- **Hardware / Environment** — Hardware used and runtime environment (GPU model, OS, etc.)
- **Runtime** — Total processing time

### 4. Open a Pull Request

Push your submission branch and open a pull request to this repository. The PR template will guide you through the required information.

### Important Notes

- **Include source code.** Your submission must contain the full source code needed to reproduce your results.
- **Specify your track.** Clearly state whether you process from raw `.bag` files or from existing `.ply` files.
- **Do not include PLY files.** Output PLY files are too large for this repository. The organizers will run your code to generate results.
- **Do not include input data.** Do not commit the dataset files themselves.

## Ranking

Results are ranked by **Chamfer Distance** (lower is better). All metrics are averaged across all sequences and frames.

### Current Rankings

| Rank | Team / Algorithm | Track | Chamfer Distance ↓ | Accuracy ↓ | Completeness ↓ | Runtime (s) |
|------|-----------------|-------|--------------------:|-----------:|----------------:|------------:|
| 1 | *Baseline* | Enhancement Only | — | — | — | — |
| | | | | | | |

> **Note:** The ranking table will be updated as submissions are evaluated. Arrows (↓) indicate that lower values are better.

## Citation

If you use this dataset, please cite the following paper:

```bibtex
@inproceedings{gautier2025uvgcwidqpc,
  author    = {Gautier, G. and Zhou, X. and Nguyen, T. and Jansen, J. and Fr{\'e}neau, L. and Viitanen, M. and Phan, U. and K{\"a}pyl{\"a}, J. and Viola, I. and Mercat, A. and Cesar, P. and Vanne, J.},
  title     = {{UVG-CWI-DQPC}: Dual-quality point cloud dataset for volumetric video applications},
  booktitle = {Proc. ACM Int. Conf. Multimedia},
  address   = {Dublin, Ireland},
  month     = oct,
  year      = {2025}
}
```

## License

Please read the [license agreement](https://ultravideo.fi/UVG-CWI-DQPC/UVG-CWI-DQPC_LICENSE_AGREEMENT.pdf) before using the dataset.

## Contact

- **Website:** [Ultra Video Group](https://ultravideo.fi/index.html)
- **GitHub:** [github.com/ultravideo](https://github.com/ultravideo)
- **Discord:** [UVG Discord](https://discord.gg/fZpub7BPUA)
