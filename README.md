# Eulerian Motion Guidance (EMG)

EMG conditions a frozen [Stable Video Diffusion (SVD-XT)][svd] backbone on
*adjacent-frame Eulerian flows* rather than the more conventional
reference-anchored Lagrangian flows. A **Bidirectional Geometric
Consistency (BGC)** loss enforces that the model's denoised latents
respect the underlying optical flow — masked by a forward-backward
cycle test that excludes occluded regions automatically. The result is
sharper motion, fewer ghosting artifacts, and substantially better
metric scores on WebVid-10M and CelebV-HQ.

## What's in the box

- Faithful PyTorch implementation of every equation in the paper:
  Eq. 4–5 (Eulerian flow), Eq. 8 (cycle energy), Eq. 9 (dynamic
  occlusion mask), Eq. 10 (geometric loss), Eq. 11–12 (parallel flow
  batching), Eq. 13 (autoregressive generative chain).
- All ablation knobs from Table 3 (motion formulation,
  consistency mode) and Table 4 (`α₁`, `α₂` sweep) plumbed through
  YAML config.
- Frozen [RAFT-Large][raft] (FlyingThings3D weights) wrapper at 256×256
  operating resolution.
- Frozen SVD-XT wrapper that loads `unet`, `vae`, `image_encoder`,
  `image_processor`, and `scheduler` from a single HF id.
- ControlNet-style FlowControlNet, U-Net Sparse-to-Dense network,
  multi-scale Motion Adapter — every trainable component initialised
  to a near-identity residual.
- Trainer with EMA shadowing, gradient clipping, mixed precision
  (bf16/fp16/fp32), gradient checkpointing, and DDP support
  (`torchrun`).
- Autoregressive long-video sampler that chains 14-frame SVD-XT
  windows up to the paper's `T = 100`.
- Full evaluation suite — LPIPS, FID, FVD, CLIP-Cons, E_warp
  (Lai *et al.* 2018), CPBD, ArcFace identity — wired to a single
  `Evaluator` that emits both JSON and Markdown reports in the
  paper's table format.
- 27 unit tests covering the core math (warping, cycle energy,
  occlusion mask, parallel-flow batching, dataloader).

[svd]: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
[raft]: https://arxiv.org/abs/2003.12039

## Repository layout

```
eulerian-motion-guidance/
├── configs/                 # YAML configurations
│   ├── default.yaml
│   ├── trajectory_webvid.yaml      # reproduces Table 1
│   └── keypoint_portrait.yaml      # reproduces Table 2
├── src/emg/
│   ├── motion/              # Eulerian flows, parallel batching, warping
│   ├── losses/              # BGC loss + diffusion surrogate
│   ├── models/              # S2D, FlowControlNet, MotionAdapter, RAFT, SVD
│   ├── data/                # WebVid + portrait loaders, sparse hints
│   ├── training/            # Trainer, EMA, schedulers
│   ├── inference/           # Eq. 13 autoregressive animator
│   ├── evaluation/          # Metric suite + report generator
│   └── utils/               # Config, logging, distributed, viz
├── scripts/                 # CLI entry points
├── tests/                   # pytest suite (CPU-only, < 1s)
├── DESIGN_NOTES.md          # 10 numbered design decisions
├── pyproject.toml           # uv-managed; py3.10–3.12
└── README.md                # this file
```

`DESIGN_NOTES.md` captures every spot where the paper does not pin
down an architectural detail and explains the choice we made.

## Installation

We ship a [uv](https://docs.astral.sh/uv/)-friendly `pyproject.toml`.

```bash
# Recommended: uv (fast, reproducible)
uv venv && source .venv/bin/activate
uv pip install -e ".[metrics,keypoints,logging,dev]"

# Or vanilla pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[metrics,keypoints,logging,dev]"
```

The `metrics` extra pulls LPIPS, clean-fid, OpenCLIP, and InsightFace.
Skip it if you only want to train.  The `keypoints` extra pulls
MediaPipe / face-alignment for portrait landmark extraction.

CUDA notes: The default torch wheels follow PyTorch's index. To pin
a specific CUDA version set `TORCH_CUDA_ARCH_LIST` and use
`uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 ...`
as appropriate.

## Pretrained weights

```bash
HF_TOKEN=hf_xxx python scripts/download_pretrained.py
```

This downloads SVD-XT through `huggingface_hub` and RAFT-Large via
`torchvision`. Both stay in your default HF / torch cache.

## Datasets

### WebVid-10M (Table 1)

The original WebVid-10M URLs were withdrawn by Shutterstock in 2024.
We do **not** redistribute videos. The loader reads a CSV manifest and
expects an MP4 directory you assemble yourself.

```bash
python scripts/download_webvid.py \
    --manifest /data/webvid/manifest.csv \
    --video-root /data/webvid/videos \
    --json-out /data/webvid/health.json
```

Manifest schema: `videoid,name,contentUrl,duration,page_dir`. Files
on disk live at `{video_root}/{page_dir}/{videoid}.mp4` (or just
`{video_root}/{videoid}.mp4`).

### Portrait — VFHQ / CelebV-HQ (Table 2)

Both are gated. Fetch them through their respective official channels:

```bash
python scripts/download_portrait.py --output ./portrait_data_info.md
```

reads out the URLs, licenses, and request procedures.

## Training

### Single-GPU

```bash
python scripts/train.py --config configs/trajectory_webvid.yaml \
    data.manifest=/data/webvid/manifest.csv \
    data.video_root=/data/webvid/videos
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node 4 scripts/train.py \
    --config configs/trajectory_webvid.yaml \
    data.manifest=/data/webvid/manifest.csv \
    data.video_root=/data/webvid/videos
```

### CPU smoke test (no real data)

```bash
python scripts/train.py --config configs/default.yaml --smoke
```

### Reproducing the ablations

The `motion.formulation`, `consistency.mode`, `consistency.alpha1`, and
`consistency.alpha2` keys map directly to Tables 3 and 4. Override
them on the CLI; no config edits required.

| Run | Override |
| --- | --- |
| Lagrangian motion (Table 3 row 1) | `motion.formulation=lagrangian` |
| No consistency loss (Table 3 row 2) | `consistency.mode=none consistency.lambda_geo=0` |
| Forward-only consistency (Table 3 row 3) | `consistency.mode=forward_only` |
| Full BGC (Table 3 row 4, default) | `consistency.mode=bidirectional` |
| Table 4 `α₁` sweep | `consistency.alpha1=0.005` (then `0.01`, `0.05`) |
| Table 4 `α₂` sweep | `consistency.alpha2=0.1` (then `0.5`, `1.0`) |

## Evaluation

```bash
python scripts/evaluate.py \
    --config configs/trajectory_webvid.yaml \
    --checkpoint checkpoints/step_00100000.pt \
    --output eval_outputs/table1.json \
    --max-samples 1000
```

Writes both `report.json` (machine-readable) and `table.md` (Markdown
table in the paper's format). Multiple runs can be merged with
`Evaluator.merge_reports([...])`.

### Tables → Configs

| Table | Config | Metrics |
| --- | --- | --- |
| Table 1 (trajectory, WebVid) | `configs/trajectory_webvid.yaml` | LPIPS, FID, FVD, CLIP-Cons, E_warp |
| Table 2 (keypoints, CelebV-HQ) | `configs/keypoint_portrait.yaml` | CPBD, ArcFace, CLIP-Cons, E_warp |
| Table 3 (ablations) | both, with overrides above | same as parent table |
| Table 4 (`α₁`, `α₂` sweep) | both, with overrides above | same as parent table |

## Single-image demo

```bash
python scripts/animate.py \
    --image inputs/portrait.png \
    --trajectories inputs/portrait_trajectories.json \
    --checkpoint checkpoints/step_00100000.pt \
    --config configs/keypoint_portrait.yaml \
    --output outputs/portrait.mp4
```

The trajectory file is JSON of the form

```json
{
  "trajectories": [
    [{"x": 64, "y": 64, "u": 1.5, "v": 0.0}],
    [{"x": 64, "y": 64, "u": 1.5, "v": 0.0}]
  ]
}
```

— one list per adjacent pair (`T-1` lists for a `T`-frame video). A
single list is broadcast to all pairs.

## Citing

```bibtex
@inproceedings{nguyen2026emg,
  title     = {Eulerian Motion Guidance: Robust Image Animation via
               Bidirectional Geometric Consistency},
  author    = {Nguyen and others},
  booktitle = {Proceedings of the ACM International Conference on
               Multimedia (MM '26)},
  year      = {2026}
}
```

## License

Apache-2.0 (see `LICENSE`).

## Acknowledgements

This implementation builds on the following community releases:

- [Stable Video Diffusion XT][svd] — Stability AI, frozen backbone.
- [RAFT][raft] — Teed & Deng, frozen flow estimator. We use the
  `torchvision` reimplementation with FlyingThings3D weights.
- [WebVid-10M][webvid] — Bain *et al.* (deprecated upstream as of 2024;
  this repo does not redistribute the data).

[webvid]: https://maxbain.com/webvid-dataset/
