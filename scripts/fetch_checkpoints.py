"""Download the fine-tuned checkpoints from the team Google Drive folder and verify them.

  python scripts/fetch_checkpoints.py              # all checkpoints
  python scripts/fetch_checkpoints.py 010 017      # only these experiments

Each file lands in checkpoints/<experiment>/model.pt (the path the experiment READMEs and model names use, e.g.
"mobileclip2-b#checkpoints/010_ft-b-contrastive/model.pt") and must match the SHA-256 below, or the script fails.
The Drive folder must be shared as "anyone with the link can view" for gdown to read it. If it is shared only with
your account, download the .pt files by hand into checkpoints/_download/ and rerun with --local.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER_ID = "10Wzied3uX7TJvHqpbQ9J9mcfdXY-AGkX"   # Drive folder "lpcvc2026-checkpoints"
SHA256 = {
    "010_ft-b-contrastive": "7edb94852271fc7693984862c006a93a6bf42860135d900e8b7a338cf936d216",   # B FT, 17.68 ms
    "012_distill-b-pecore": "2d38af83516a5df543408c7078402bc04566b33ed7eb4a68e2edfdf46cdc5751",   # ablation
    "013_ft-b-crosssource": "ea330bea691e9e01e53bb88adb77cd0549ce4eaa4550fde372f87b97ad610648",   # ablation
    "015_ft-s0-contrastive": "3ac0a0f18992c7d99fce90aac58026c920081033d2e2794d9a5631178da8cc2c",  # S0 FT, 10.05 ms
    "017_ft-s3-contrastive": "46d94527398e5a7b2958a6947c832855c0f3e7460a92d60e7b6960a31f590246",  # S3 FT, 28.35 ms
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="experiment number prefixes, e.g. 010 017")
    ap.add_argument("--local", action="store_true", help="skip the download; use files already in checkpoints/_download/")
    args = ap.parse_args()
    want = [k for k in SHA256 if not args.only or k[:3] in args.only]
    dl = ROOT / "checkpoints" / "_download"
    todo = [k for k in want if not ((ROOT / "checkpoints" / k / "model.pt").exists()
                                    and sha256(ROOT / "checkpoints" / k / "model.pt") == SHA256[k])]
    if not todo:
        print("all requested checkpoints present and verified")
        return
    if not args.local:
        import gdown

        dl.mkdir(parents=True, exist_ok=True)
        files = gdown.download_folder(id=FOLDER_ID, output=str(dl), quiet=False, skip_download=True)
        for f in files:
            name = Path(f.path).stem
            if name in todo:
                gdown.download(id=f.id, output=str(dl / f"{name}.pt"), quiet=False)
    bad = []
    for k in todo:
        src = dl / f"{k}.pt"
        if not src.exists():
            bad.append(f"{k}: not downloaded"); continue
        if sha256(src) != SHA256[k]:
            bad.append(f"{k}: SHA-256 mismatch"); continue
        dst = ROOT / "checkpoints" / k / "model.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst)
        print(f"ok  {dst.relative_to(ROOT).as_posix()}")
    if bad:
        sys.exit("FAILED: " + "; ".join(bad))


if __name__ == "__main__":
    main()
