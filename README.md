# On-device image–text retrieval: beating the LPCVC 2026 Track 1 baseline

CAP 6411 (Fall 2026), Topic 18. Rafael Fernandes da Silva, Erick Castillo, Carl Augustin.

**Goal:** higher fractional Recall@10 than the official sample (OpenAI CLIP ViT-B/16) with image + text encoder
latency under 35 ms on **Samsung Galaxy S22 (Family)** via Qualcomm AI Hub. Every design decision is backed by an
on-device measurement.

Track rules: https://lpcv.ai/2026LPCVC/tracks/track1 (the XR2 Gen 2 proxy used in the sample code was deprecated
in June 2026; the open submission targets the S22).

## Setup

```bash
git clone --recurse-submodules <this repo> && cd lpcvc2026
conda create -p ./.venv python=3.11 -y          # or python -m venv .venv
.venv/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv/python -m pip install -r requirements.txt
.venv/python -m gdown --folder https://drive.google.com/drive/folders/1tTwrehwLtVtMOTjD5beYDC3lcWIfzS5D -O data/lpcvc_sample
.venv/python scripts/build_val_set.py              # RefCOCOg-LPCVC validation: 5 folds x 300 images (~250 MB)
.venv/python scripts/build_train_set.py            # RefCOCO/+/g train, LPCVC-style: 15.9k images, 76k captions (~2.5 GB)
.venv/python scripts/build_coco_captions_set.py    # same images, COCO captions (control; ~12 MB fetched)
.venv/python scripts/build_grouped_val_set.py      # LPCVC-like multi-instance validation: 4 folds x 300 images
.venv/python scripts/build_test_set.py             # held-out test fold: 300 RefCOCO/+ images (~50 MB)
.venv/python scripts/fetch_checkpoints.py 010 015 017   # fine-tuned B/S0/S3 from the team Drive folder (SHA-256 checked)
& .\scripts\configure_aihub.ps1                   # stores your AI Hub token (hidden prompt)
.venv/python -m pytest -q tests
```

## Workflow

| Step | Command | Where numbers come from |
|---|---|---|
| Screen models locally | `python scripts/eval_local.py --exp <slug> --title ... --question ... --models clip-b16 mobileclip2-s0` | RTX GPU, PyTorch. RefCOCOg-LPCVC (pooled folds) + sample. |
| Export + ONNX parity | `python scripts/export_onnx.py --model mobileclip2-s0` | Checks ONNX vs PyTorch on real inputs. |
| Fine-tune a student | `python scripts/train.py --exp <slug> --student mobileclip2-b [--teacher pe-core-l14-336 --alpha 0.5]` | Writes `checkpoints/NNN_<slug>/model.pt`; last epoch is the result. |
| Screen teachers | `python scripts/screen_teachers.py --exp <slug> --teachers ... --ensembles a+b` | Zero-shot, large open_clip models (never deployed). |
| Measure on device | `python scripts/aihub_run.py --exp <slug> --models clip-b16 mobileclip2-s0 --profile-runs 3` | S22, QNN. **The only latency we report.** |

### Experiments are numbered and self-describing
Every run of `eval_local.py` / `aihub_run.py` writes `experiments/NNN_<slug>/`:
`README.md` (question, exact reproduce command, results table, **Conclusions written by us**), `manifest.json`
(command, git commit, dirty flag, versions, GPU, args), `log.txt` (full console output) and raw results
(per-image recalls, AI Hub job IDs, downloaded profiles, device embeddings). `experiments/INDEX.md` lists them all in
order. Rules: commit code before running (the manifest records the commit), use a new slug per question, fill in
*Conclusions* right after a run, and never delete failed experiments.

Model names (see `lpcvc/models.py`): `clip-b16` (official baseline as shipped), `clip-b16-norm`, `mobileclip2-s0`,
`mobileclip2-s2`, `mobileclip2-b`, `mobileclip2-s3`, `mobileclip2-s4`; add `@<res>` for a different internal image
resolution, e.g. `mobileclip2-s0@192`; add `#<checkpoint>` for fine-tuned weights and `~<a>` to blend them with the
pretrained weights, e.g. `"mobileclip2-b#checkpoints/010_ft-b-contrastive/model.pt"`. Every script accepts these names.
Graph flags go before `@`/`#` and keep the weights: `+fmask` (finite −100 causal mask, needed to quantize B's text
tower), `+gelutanh`, `+gelusig` (QuickGELU, 12% faster on the S22), e.g.
`"mobileclip2-b+gelusig#checkpoints/010_ft-b-contrastive/model.pt"` is the QuickGELU variant of fine-tuned B (exp 023).

### Ground rules
- Every encoder takes the **official interface** (image `1x3x224x224` float in [0,1], no normalization; text `1x77`
  int, CLIP tokenizer with EOT padding). Normalization, resizing and token re-padding live inside the graph, so they
  are counted in latency.
- `lpcvc/metrics.py` is tested against a verbatim copy of the official scorer (`tests/test_metrics.py`).
- The LPCVC sample (56 images) is **never** used for training, calibration or model selection. With 56 queries the
  95% CI is about ±6 points. **Select on RefCOCOg-LPCVC** (1,500 queries, about ±2 points; see
  `scripts/build_val_set.py` for how it was filtered and normalized to the LPCVC caption style) and report the
  sample as a held-out check.
- Both validation sets (`refcocog`, `grouped`) use RefCOCO-family captions, so they **inflate gains from training on
  RefCOCO-family data** (experiments 020, 021: +22 on grouped vs +2.2 on the sample). Use them to rank variants trained on
  the same data, not to choose the training data.
- **The test fold (`data/refcoco_test`, from `scripts/build_test_set.py`) is scored once, for the final report.**
  It shares no image with training, validation or calibration. Nothing is selected or tuned on it, and
  `eval_local.py` refuses it unless `--final` is passed.
- **Never train or calibrate on images in `splits/reserved_coco_train2014_ids.txt`** (all RefCOCOg val/test images,
  which are COCO train2014). Filter COCO, RefCOCO/RefCOCO+ and Visual Genome training data against this list.
  Those datasets share COCO train2014 images.
- Compile options match the sample: `--target_runtime qnn_dlc --truncate_64bit_io`.

## Layout
```
lpcvc/          data loading (official preprocessing), metrics, model wrappers, teachers, experiment registry
scripts/        build_val_set.py, build_grouped_val_set.py, build_test_set.py, build_train_set.py, build_coco_captions_set.py, build_calib_set.py, eval_local.py, subsample_pool.py, train.py,
                screen_teachers.py, export_onnx.py, aihub_run.py, aihub_quantize.py, plot_pareto.py
splits/         committed validation and test folds, training/calibration image lists, reserved (never-train) COCO IDs
checkpoints/    fine-tuned weights (gitignored, not on GitHub; scripts/fetch_checkpoints.py downloads them from Drive)
tests/          scorer equivalence tests
experiments/    INDEX.md + one numbered folder per experiment (see above)
docs/           project review (2026-11-17), presentation slides (HTML source + 16:9 PDF)
third_party/    lpcvc_sample (official sample solution), ml-mobileclip (submodules, unmodified)
```

## Credits
Official sample: github.com/lpcvai/26LPCVC_Track1_Sample_Solution. MobileCLIP2 weights via OpenCLIP
(github.com/apple/ml-mobileclip, Faghri et al. 2025). CLIP (Radford et al. 2021). TinyCLIP (Wu et al. 2023).
