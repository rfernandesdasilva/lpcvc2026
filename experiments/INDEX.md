# Experiment index

In the order they were run. Each folder has a README (question, how to reproduce, results, conclusions), `manifest.json` (exact command, commit, versions) and `log.txt`.

| ID | Date | Experiment | Status | Headline | Commit |
|---|---|---|---|---|---|
| [001](001_local-screen-sample/README.md) | 2026-11-17 | Baseline reproduction and compact-model screen on the LPCVC sample | ok | best on sample: mobileclip2-s2 89.45 | `17823c1` |
| [002](002_local-screen-refcocog/README.md) | 2026-11-17 | Model screen on the RefCOCOg-LPCVC validation set (5x300) | ok | best on refcocog: mobileclip2-b 71.31 | `17823c1` |
| [003](003_s22-smoke-test/README.md) | 2026-11-17 | AI Hub pipeline smoke test on the S22 | ok | best valid by RefCOCOg: mobileclip2-s0 (10.1 ms) | `17823c1` |
