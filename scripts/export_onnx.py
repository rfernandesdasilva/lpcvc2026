"""Export image/text encoders to ONNX with the official input interface and check parity with PyTorch.

  python scripts/export_onnx.py --model mobileclip2-s0
  -> exported/mobileclip2-s0/{image_encoder.onnx,text_encoder.onnx}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpcvc.data import load_lpcvc_sample  # noqa: E402
from lpcvc.models import build  # noqa: E402


def export(model: str, out_dir: Path, opset: int = 18) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_enc, txt_enc = build(model, "cpu")
    img_path, txt_path = out_dir / "image_encoder.onnx", out_dir / "text_encoder.onnx"
    dummy_img = torch.rand(1, 3, 224, 224)
    dummy_txt = torch.randint(0, 49408, (1, 77), dtype=torch.int64)  # int64 + --truncate_64bit_io, as in the sample
    with torch.no_grad():
        torch.onnx.export(img_enc, (dummy_img,), str(img_path), input_names=["image"], output_names=["embedding"],
                          opset_version=opset, do_constant_folding=True, dynamo=False)
        torch.onnx.export(txt_enc, (dummy_txt,), str(txt_path), input_names=["text"], output_names=["text_embedding"],
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    return img_path, txt_path


def parity(model: str, img_path: Path, txt_path: Path, data: Path, n: int = 8) -> None:
    import onnxruntime as ort

    ds = load_lpcvc_sample(data)
    imgs, toks = ds.images()[:n], ds.tokens()[:n]
    img_enc, txt_enc = build(model, "cpu")
    si = ort.InferenceSession(str(img_path), providers=["CPUExecutionProvider"])
    st = ort.InferenceSession(str(txt_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        for i in range(n):
            a = img_enc(torch.from_numpy(imgs[i:i + 1])).numpy()
            b = si.run(None, {"image": imgs[i:i + 1]})[0]
            c = txt_enc(torch.from_numpy(toks[i:i + 1])).numpy()
            d = st.run(None, {"text": toks[i:i + 1].astype(np.int64)})[0]
            ci, ct = float((a * b).sum()), float((c * d).sum())
            assert ci > 0.9999 and ct > 0.9999, f"parity failed on sample {i}: img cos={ci}, txt cos={ct}"
    print(f"ONNX parity OK on {n} real samples (cosine > 0.9999)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=str(ROOT / "data" / "lpcvc_sample"))
    args = ap.parse_args()
    out = Path(args.out or ROOT / "exported" / args.model.replace("@", "_"))
    ip, tp = export(args.model, out)
    for p in (ip, tp):
        size = sum(f.stat().st_size for f in p.parent.glob(p.stem + "*")) / 2**20
        print(f"{p}  {size:.1f} MiB")
    parity(args.model, ip, tp, Path(args.data))


if __name__ == "__main__":
    main()
