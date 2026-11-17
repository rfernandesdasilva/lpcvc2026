# Experiment log

Newest first. One row per run; link the JSON in `results/runs/`. Record failures too.

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
