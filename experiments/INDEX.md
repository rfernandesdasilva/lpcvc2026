# Experiment index

In the order they were run. Each folder has a README (question, how to reproduce, results, conclusions), `manifest.json` (exact command, commit, versions) and `log.txt`.

| ID | Date | Experiment | Status | Headline | Commit |
|---|---|---|---|---|---|
| [001](001_local-screen-sample/README.md) | 2026-11-17 | Baseline reproduction and compact-model screen on the LPCVC sample | ok | best on sample: mobileclip2-s2 89.45 | `17823c1` |
| [002](002_local-screen-refcocog/README.md) | 2026-11-17 | Model screen on the RefCOCOg-LPCVC validation set (5x300) | ok | best on refcocog: mobileclip2-b 71.31 | `17823c1` |
| [003](003_s22-smoke-test/README.md) | 2026-11-17 | AI Hub pipeline smoke test on the S22 | ok | best valid by RefCOCOg: mobileclip2-s0 (10.1 ms) | `17823c1` |
| [004](004_s22-profile-candidates/README.md) | 2026-11-17 | S22 latency and on-device accuracy of all candidate models | ok | best valid by RefCOCOg: mobileclip2-b (17.7 ms) | `a65b584` |
| [005](005_local-screen-large/README.md) | 2026-11-17 | Local accuracy of the larger MobileCLIP2 models (S3, S4) | ok | best on refcocog: mobileclip2-s3 72.11 | `4ff115d` |
| [006](006_s22-profile-large/README.md) | 2026-11-17 | S22 latency of the larger MobileCLIP2 models (S3, S4) | ok | best valid by RefCOCOg: mobileclip2-s3 (28.4 ms) | `4ff115d` |
| [007](007_s22-int8-ptq/README.md) | 2026-11-17 | INT8 post-training quantization of MobileCLIP2-B and S0 on the S22 | failed | failed (zip download bug); quantize jobs reused by 008 | `ed5e0df` |
| [008](008_s22-int8-ptq-rerun/README.md) | 2026-11-17 | INT8 post-training quantization of MobileCLIP2-B and S0 on the S22 (rerun of 007) | partial | default AI Hub PTQ fails: w8a8 destroys accuracy; most INT8 graphs fail on S22; B img w8a8 only 5% faster | `f29ec23` |
| [009](009_teacher-screen/README.md) | 2026-11-17 | Teacher screening (zero-shot, local) | ok | best teacher on refcocog: mobileclip2-l14+vit-h14-378-dfn5b+pe-core-l14-336 74.19 | `b5607aa` |
| [010](010_ft-b-contrastive/README.md) | 2026-11-17 | Fine-tune MobileCLIP2-B, contrastive only (control) | ok | mobileclip2-b refcocog: 71.38 -> 87.34 (+15.96) | `b5607aa` |
| [011](011_s22-finetuned-b/README.md) | 2026-11-17 | Fine-tuned MobileCLIP2-B on the Galaxy S22 | ok | best valid by RefCOCOg: mobileclip2-b#checkpoints/010_ft-b-contrastive/model.pt (17.7 ms) | `b5607aa` |
| [012](012_distill-b-pecore/README.md) | 2026-11-17 | Fine-tune MobileCLIP2-B with distillation from PE-Core-L/14-336 (alpha 0.5) | ok | mobileclip2-b refcocog: 71.38 -> 81.74 (+10.36) | `b5607aa` |
| [013](013_ft-b-crosssource/README.md) | 2026-11-17 | Fine-tune MobileCLIP2-B on RefCOCO/RefCOCO+ only (cross-source ablation) | ok | mobileclip2-b refcocog: 71.38 -> 84.36 (+12.98) | `b5607aa` |
| [014](014_wiseft-b/README.md) | 2026-11-17 | WiSE-FT weight blending of fine-tuned B | ok | best on refcocog: mobileclip2-b#checkpoints/010_ft-b-contrastive/model.pt 87.28 | `b5607aa` |
| [015](015_ft-s0-contrastive/README.md) | 2026-11-17 | Fine-tune MobileCLIP2-S0, contrastive (same recipe as 010) | ok | mobileclip2-s0 refcocog: 69.21 -> 82.24 (+13.03) | `b5607aa` |
| [016](016_s22-finetuned-s0/README.md) | 2026-11-17 | Fine-tuned MobileCLIP2-S0 on the Galaxy S22 | ok | best valid by RefCOCOg: mobileclip2-s0#checkpoints/015_ft-s0-contrastive/model.pt (10.1 ms) | `23a60bc` |
| [017](017_ft-s3-contrastive/README.md) | 2026-11-17 | Fine-tune MobileCLIP2-S3, contrastive (same recipe as 010) | ok | mobileclip2-s3 refcocog: 72.08 -> 87.95 (+15.87) | `5fceb28` |
| [018](018_s22-finetuned-s3/README.md) | 2026-11-17 | Fine-tuned MobileCLIP2-S3 on the Galaxy S22 | ok | best valid by RefCOCOg: mobileclip2-s3#checkpoints/017_ft-s3-contrastive/model.pt (28.4 ms) | `5fceb28` |
| [019](019_compare-finetuned/README.md) | 2026-11-17 | Fine-tuned students head to head (FP32) | ok | best on refcocog: mobileclip2-s3#checkpoints/017_ft-s3-contrastive/model.pt 88.01 | `5fceb28` |
| [020](020_ft-b-cococaptions/README.md) | 2026-11-18 | Fine-tune MobileCLIP2-B on COCO Captions (caption-type control) | ok | mobileclip2-b refcocog: 71.38 -> 75.69 (+4.31) | `3c2fa78` |
