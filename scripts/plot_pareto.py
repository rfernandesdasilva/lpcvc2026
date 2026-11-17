"""Accuracy-latency plot for an AI Hub experiment: S22 total latency vs local RefCOCOg-LPCVC frac R@10.

  python scripts/plot_pareto.py experiments/004_s22-profile-candidates [more experiment dirs ...]
  -> pareto.png in the LAST experiment dir (points from all given experiments are combined)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def label(m: str) -> str:
    """Short point label: 'mobileclip2-b#checkpoints/010_ft-b-contrastive/model.pt' -> 'mobileclip2-b FT (010)'."""
    if "#" not in m:
        return m
    base, ck = m.split("#", 1)
    return f"{base} FT ({ck.split('/')[1][:3]})"


def main(exp_dirs: list[Path]):
    exp_dir = exp_dirs[-1]
    rows = [r for d in exp_dirs for r in json.loads((d / "results.json").read_text(encoding="utf-8"))["rows"]
            if "error" not in r and r.get("local_refcocog")]
    pts = sorted(((r["total_ms"], r["local_refcocog"][0] * 100, r["model"]) for r in rows))
    front, best = [], -1.0
    for x, y, m in pts:
        if y > best:
            front.append((x, y)); best = y
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.axvspan(35, max(40, max(p[0] for p in pts) + 5), color="0.92")
    ax.axvline(35, color="0.5", lw=1, ls="--")
    ax.text(35.4, max(p[1] for p in pts), "over\nbudget", color="0.4", fontsize=8, va="top")
    ax.plot(*zip(*front), color="#1f6feb", lw=1.2, zorder=1, label="Pareto front")
    for x, y, m in pts:
        base, ft = m.startswith("clip"), "#" in m
        ax.scatter(x, y, s=36, zorder=2, color="#9a6700" if base else ("#cf222e" if ft else "#1f6feb"),
                   marker="s" if base else ("D" if ft else "o"))
        ax.annotate(label(m), (x, y), textcoords="offset points", xytext=(5, -3), fontsize=7.5)
    ax.set_xlabel("Image + text encoder latency on Galaxy S22, QNN (ms)")
    ax.set_ylabel("Fractional Recall@10, RefCOCOg-LPCVC (%)")
    ax.set_xlim(0, max(40, max(p[0] for p in pts) + 5))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out = exp_dir / "pareto.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
