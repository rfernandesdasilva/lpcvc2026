# Experiment log

Newest first. One row per run; link the JSON in `results/runs/`. Record failures too.

## Local screening: RefCOCOg-LPCVC validation (5 folds × 300 images), RTX 4080 SUPER, FP32, 2026-11-17

Run: `runs/20261117_144736_local.json`. Validation set built by `scripts/build_val_set.py`. Each fold has its own
pool of about 690 captions, and the results pool all 1,500 query images. The 95% CIs are bootstraps over images,
and Δ is a paired bootstrap against `clip-b16`. `*` = CI excludes 0.

| Model | RefCOCOg-LPCVC frac R@10 | 95% CI | fold range | Δ vs clip-b16 | LPCVC sample (n=56) |
|---|---|---|---|---|---|
| clip-b16 (official) | 63.19 | 61.2–65.2 | 61.0–67.8 | — | 87.35 |
| clip-b16-norm | 64.72 | 62.7–66.8 | 63.3–68.2 | +1.53* (+0.5…+2.6) | 88.85 |
| mobileclip2-s0 (256) | 69.17 | 67.3–71.0 | 68.0–72.4 | +5.98* (+4.4…+7.6) | 88.85 |
| mobileclip2-s0@224 | 68.69 | 66.7–70.6 | 67.1–72.2 | +5.50* (+3.8…+7.1) | 88.82 |
| mobileclip2-s0@192 | 66.58 | 64.6–68.5 | 65.2–70.2 | +3.39* (+1.6…+5.1) | 89.00 |
| mobileclip2-s2 | 71.15 | 69.4–73.0 | 69.9–74.6 | +7.96* (+6.4…+9.6) | 89.45 |
| mobileclip2-b | 71.31 | 69.5–73.1 | 69.8–73.4 | +8.12* (+6.6…+9.7) | 89.21 |

Observations:
- With 1,500 queries the CI is about ±2 points (vs. ±6 on the sample), and every model separates from the baseline.
  The ranking is S2 ≈ B > S0 > S0@224 > S0@192 > CLIP-norm > CLIP. None of these gaps were visible on the sample.
- Lowering S0's internal resolution costs 0.5 points at 224 and 2.6 points at 192. Whether that buys real latency
  must be measured on the S22. FastViT's conv stages scale with pixels, but the S22 profile decides.
- The leaderboard ranks by accuracy *among valid (<35 ms) models*. MobileCLIP2-B has a ViT-B/16 image tower like
  the baseline (baseline on S22: 17.7 + 4.6 = 22.2 ms), so B and S2 may fit the budget with 2 points more than S0.
  Profile them on the S22 before committing to S0.
- The validation set is a proxy. Its captions are longer (7.2 vs 5.0 words) and fewer per image (2.3 vs 4.1) than
  LPCVC's. Absolute numbers aren't comparable to the sample; use it to rank and select only.

## Local screening: LPCVC sample (56 images / 211 captions), RTX 4080 SUPER, PyTorch FP32, 2026-11-17

Run: `runs/20261117_143052_local.json`. 95% CIs are bootstrap over the 56 query images.

| Model | frac R@10 | 95% CI | Hit@10 | Hit@1 | Δ vs clip-b16 (paired 95% CI) |
|---|---|---|---|---|---|
| clip-b16 (official, no input norm) | 87.35 | 80.6–93.4 | 98.21 | 75.00 | — |
| clip-b16-norm | 88.85 | 82.2–94.7 | 98.21 | 76.79 | +1.50 (−0.2 … +3.5) |
| mobileclip2-s0 (256 in-graph) | 88.85 | 82.6–94.5 | 98.21 | 80.36 | +1.50 (−1.1 … +4.0) |
| mobileclip2-s0@224 | 88.82 | 82.5–94.6 | 96.43 | 80.36 | +1.47 (−1.5 … +4.4) |
| mobileclip2-s2 | 89.45 | 83.7–94.8 | 98.21 | 82.14 | +2.10 (−0.2 … +4.6) |
| mobileclip2-b | 89.21 | 83.2–94.8 | 98.21 | 80.36 | +1.86 (−0.3 … +4.2) |

Observations:
- Local CLIP baseline (87.35) is close to the teammate's S22 measurement (86.97). The small gap is plausibly QNN
  numerics, but that is not yet verified.
- The official sample feeds [0,1] pixels to CLIP **without** mean/std normalization. Adding it gives +1.5 points.
  So a "fixed baseline" should be reported next to the official one.
- **No difference is significant on 56 images.** We need a larger validation set that resembles the LPCVC caption
  style (short object-attribute descriptions, several per image) before choosing among models or ablations.
- ONNX export of mobileclip2-s0: parity cos > 0.9999 on 8 real samples. Sizes: image 43.4 MiB, text 242.1 MiB (FP32).
  The text tower is 85% of the bytes.

## Device (S22, QNN): from the teammate's notebook (TFLite), not yet reproduced with `aihub_run.py`

| Model | frac R@10 | img ms | txt ms | total ms |
|---|---|---|---|---|
| CLIP ViT-B/16 | 86.97 | 17.655 | 4.591 | 22.246 |
| MobileCLIP2-S0 | 89.40 | 6.384 | 3.314 | 9.698 |
