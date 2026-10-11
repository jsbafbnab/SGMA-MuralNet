# SGMA-MuralNet

**Structure-Guided Texture Enhancement and Multi-Scale Feature Fusion for Progressive Mural Inpainting**

Implementation of **SGMA-MuralNet**, a two-stage network for mural inpainting that integrates structure-conditioned deep feature modulation and multi-scale feature aggregation (MFA) into MuralNet's progressive SRN–CCN framework.

---

## Abstract

Progressive mural inpainting requires structural guidance to reconstruct missing textures while maintaining color consistency. Existing methods only introduce line constraints at the input stage, resulting in distorted structures and color mismatch for severely damaged murals. SGMA-MuralNet is a two-stage network that (i) applies affine modulation on channel-local normalized features via a dedicated full-line encoder to retain fine mural structures, (ii) uses parallel dilated convolutions and channel recalibration (MFA) to capture multi-range context, and (iii) predicts a signed residual for color alignment with mask compositing to keep intact original pixels. On DhMurals1714 with 10%–60% irregular damage masks, our model reaches **28.65 dB PSNR, 0.926 SSIM and 0.034 LPIPS**, outperforming MuralNet by **+1.37 dB PSNR** and a **30.6% LPIPS reduction**.

---

## Quantitative results on DhMurals1714

1250 image–mask evaluation pairs (5 area bins × 5 masks × 50 test images). Full PSNR / SSIM / LPIPS / Edge F1 / hole PSNR.

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Edge F1↑ | Hole PSNR↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| *Image-only input (damaged image + mask)* | | | | | |
| LaMa | 25.42 | 0.865 | 0.089 | 0.594 | 20.44 |
| MAT | 25.88 | 0.874 | 0.081 | 0.612 | 20.90 |
| MISF | 26.05 | 0.879 | 0.076 | 0.628 | 21.07 |
| ZITS | 26.25 | 0.882 | 0.072 | 0.665 | 21.27 |
| SEM-Net | 26.43 | 0.887 | 0.067 | 0.681 | 21.45 |
| *Complete-line guided (damaged image + line drawing + mask)* | | | | | |
| EdgeConnect | 26.48 | 0.893 | 0.059 | 0.785 | 21.50 |
| MuralNet | 27.28 | 0.902 | 0.049 | 0.823 | 22.30 |
| StrDiffusion | 27.71 | 0.913 | 0.043 | 0.841 | 22.73 |
| **SGMA-MuralNet (ours)** | **28.65** | **0.926** | **0.034** | **0.886** | **23.67** |

Per-bin results are reported in Table 8 of the paper; robustness to line-guidance perturbations in Table 12.

---

## Architecture

```
              ┌─────────────────────── Stage 1: SRN ───────────────────────┐
 Io, L ──────►│ [Io,L]→stem→T0 → T1 → Z2→Mod2→T2 → Z3→Mod3→T3             │
              │                        ▲        ▲                          │
 L ──────────►│ LineEncoder ───────────┴────────┘  (S2, S3 → a_i, b_i)     │
              │ T3 → ResBlocks×4 → MFA → Fout → skip decoder → Js          │
              └────────────────────────────────────────────────────────────┘
              bIs = Io + M ⊙ Js

              ┌─────────────────────── Stage 2: CCN ───────────────────────┐
 [bIs, M] ───►│ U-Net (+self-attention) → 7×7 conv + tanh → R ∈ [-1,1]     │
              └────────────────────────────────────────────────────────────┘
              Jf = clip(bIs + R, 0, 1),   bI = Io + M ⊙ Jf
```

The SRN integrates a complete-line encoder, structure-conditioned modulation at
the quarter/eighth encoder scales, and a bottleneck MFA adapter (four dilated
branches {1,3,5,7} + squeeze-and-excitation + zero-initialized 1×1 projection).
The CCN performs residual color refinement to produce the final restored mural.

---

## Installation

```bash
# with pip
pip install -r requirements.txt

# or with conda
conda env create -f environment.yml
conda activate sgma-muralnet
```

Requirements: Python 3.10+, PyTorch ≥ 2.1. LPIPS evaluation requires the
`lpips` package; MAC counting requires `thop` (both in `requirements.txt`).

---

## Dataset: DhMurals1714

Download: https://pan.baidu.com/s/1AQRhOkErkGzRZ5w3_BsoQQ?pwd=wj5i

```
DATA_ROOT/
  train/{images,edges,masks}/     # 1564 images, 1564 edges, 5000 masks (pool)
  val/{images,edges,masks}/       # 100 images, 100 edges, 2000 masks (pool)
  test/{images,edges,masks}/      # 50 images, 50 edges, 50 masks (1:1)
```

- `images/*.jpg` — mural photographs (RGB).
- `edges/*.jpg`  — registered line drawings (grayscale, dark lines on white;
  the loader inverts so that line pixels == 1).
- `masks/*.png`  — binary masks, white = missing/hole.

Set `DATA_ROOT` in `config/default.yaml` or pass `--data-root`. Alternatively
provide `.flist` files and generate them with:

```bash
python -m scripts.make_flist --data-root <DATA_ROOT> --out <FLIST_DIR>
```

### Cross-partition geometric screening (Section 4.1)

Screens every cross-partition image pair (train-test, train-val, val-test) for
overlapping content using SIFT descriptors (max 5000 keypoints), FLANN
2-nearest-neighbor matching, Lowe's ratio at 0.75 and RANSAC homography with a
3.0-pixel reprojection threshold; pairs with ≥20 inliers are flagged:

```bash
python -m scripts.sift_screening --data-root <DATA_ROOT> --workers 8   # full (239,600 pairs)
python -m scripts.sift_screening --data-root <DATA_ROOT> --limit 20    # quick test
```

---

## Training

Two-phase progressive training. Phase 1 trains the SRN; Phase 2 initializes the
CCN from the best Phase-1 checkpoint and trains the full pipeline.

```bash
# Phase 1 (SRN only)
python train.py --phase 1 --data-root <DATA_ROOT>

# Phase 2 (SRN + CCN)
python train.py --phase 2 --data-root <DATA_ROOT>
```

Key options: `--variant A0..A8` (ablation, default `A3`), `--seed N`,
`--resume <ckpt>`, `--device cuda`, `--output-dir <dir>`, `--config <yaml>`.
All hyperparameters are in `config/default.yaml`.

Run the three-seed protocol in one command:

```bash
python -m scripts.run_multiseed --data-root <DATA_ROOT> [--variant A3]
```

---

## Testing & evaluation

### Inference

```bash
python test.py --checkpoint <CKPT> --data-root <DATA_ROOT> --out-dir restored
```

### Benchmark evaluation (1250 pairs, Tables 7–8)

```bash
python -m sgma_muralnet.evaluate --checkpoint <CKPT> --data-root <DATA_ROOT>
```

Reports full/hole PSNR, SSIM, LPIPS, Edge F1 (overall and per area bin).

### Guidance-perturbation robustness (Table 12)

```bash
python -m sgma_muralnet.evaluate --checkpoint <CKPT> --data-root <DATA_ROOT> --perturb
```

### Paired bootstrap vs baseline (Table 9)

```bash
python -m sgma_muralnet.evaluate --checkpoint <OURS_CKPT> \
    --baseline-checkpoint <A0_CKPT> --bootstrap
```

### Efficiency (Table 15)

```bash
python -m sgma_muralnet.benchmark [--variant A3] [--device cuda]
```

### Affine-parameter visualization (Figure 6)

```bash
python -m sgma_muralnet.visualize --checkpoint <CKPT> \
    --data-root <DATA_ROOT> --image-index 0 --out-dir visualizations
```

### Architecture self-check

```bash
python -m scripts.verify_architecture
```

### One-command smoke test

Runs every entry point (train → test → evaluate → benchmark → visualize, plus
the utility scripts) on a tiny synthetic dataset and reports pass/fail:

```bash
python -m scripts.smoke_test
```

---

## Ablation variants (Table 10)

`--variant` selects one of:

| Variant | Configuration | | Variant | Configuration |
| --- | --- | --- | --- | --- |
| A0 | baseline MuralNet | | A5 | direct concat + 1×1 (control) |
| A1 | + structural modulation | | A6 | matched-capacity conv (control) |
| A2 | + bottleneck MFA | | A7 | combined global+local norm |
| A3 | **full model (default)** | | A8 | modulation without input concat |
| A4 | local norm only (control) | | | |

Sensitivity controls (Table 11): `MODULATION_SCALES: [2]` / `[3]`,
`DILATIONS: [1,2,3,4]`, `USE_SE: false`.

---

## Pretrained models

The full trained checkpoints will be released after the paper is accepted.
For training from scratch, checkpoints are saved to `OUTPUT_DIR`; the best
validation-hole-MSE model is kept as `phase1_best.pt` and `phase2_best.pt`,
and loaded with `--checkpoint` in the `test.py` / `evaluate` / `visualize`
commands above.

---

## Acknowledgements

This work builds on [MuralNet](https://github.com/qinnzou/mural-image-inpainting)
and the DhMurals1714 benchmark.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Repository structure

```
SGMA-MuralNet/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── .gitignore
├── train.py                         # top-level entry: python train.py
├── test.py                          # top-level entry: python test.py
├── config/
│   └── default.yaml                 # hyperparameters
├── sgma_muralnet/
│   ├── __init__.py
│   ├── config.py                    # YAML config loader
│   ├── model.py                     # SGMAMuralNet (SRN + CCN)
│   ├── dataset.py                   # DhMurals1714 loader
│   ├── metrics.py                   # PSNR / SSIM / LPIPS
│   ├── utils.py                     # seed / checkpoint / image helpers
│   ├── train.py                     # two-phase training
│   ├── test.py                      # inference / eval
│   ├── evaluate.py                  # 1250-pair / perturbations / bootstrap
│   ├── benchmark.py                 # MACs / latency / memory
│   ├── masks.py                     # free-form mask generation + area bins
│   ├── perturbations.py             # line-guidance perturbations
│   ├── edge_f1.py                   # edge F1 metric
│   ├── bootstrap.py                 # paired bootstrap
│   ├── visualize.py                 # affine-parameter heatmaps
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── line_encoder.py          # complete-line encoder
│   │   ├── modulation.py            # channel-local norm + conditioning heads
│   │   ├── mfa.py                   # dilated branches + SE + projection
│   │   ├── srn.py                   # structure reconstruction network
│   │   ├── ccn.py                   # color correction network
│   │   ├── discriminator.py         # 70x70 PatchGAN
│   │   └── modules.py               # residual block / self-attention / init
│   └── losses/
│       ├── __init__.py
│       ├── vgg.py                   # VGG-19 feature extractor
│       ├── perceptual.py            # content / style losses
│       ├── adversarial.py           # relativistic average loss
│       ├── histogram.py             # feature histogram matching loss
│       └── criterion.py             # combined training objective
└── scripts/
    ├── verify_architecture.py       # parameter / shape self-check
    ├── run_multiseed.py             # three-seed training
    ├── make_flist.py                # generate .flist files
    ├── sift_screening.py            # SIFT cross-partition screening
    └── smoke_test.py                # one-command end-to-end smoke test
```

---

## References

1. Li et al., *Line Drawing-Guided Progressive Inpainting for Mural Damage*, JOCCH 2025.
2. Liu et al., *Thinking inside the Convolution for Image Inpainting* (TSGL), arXiv 2026.
3. Zamir et al., *Multi-Stage Progressive Image Restoration* (MPR-Net), CVPR 2021.
4. Hu et al., *Squeeze-and-Excitation Networks* (SENet), CVPR 2018.
5. Johnson et al., *Perceptual Losses for Real-Time Style Transfer*, ECCV 2016.
