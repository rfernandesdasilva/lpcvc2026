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
| [008](008_s22-int8-ptq-rerun/README.md) | 2026-11-17 | INT8 post-training quantization of MobileCLIP2-B and S0 on the S22 (rerun of 007) | running |  | `f29ec23` |
