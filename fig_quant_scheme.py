#!/usr/bin/env python3
"""
Figura didatica: quantizacao dos pesos por tensor x por canal, com pesos reais.

Usa uma camada depthwise da EfficientNet-Lite0 treinada (BN fundida, como na quantizacao real)
e mostra tres filtros da mesma camada: o de maior amplitude, um mediano e um pequeno. Para cada
um, os 9 pesos em FP32 e os valores INT8 com uma escala para a camada inteira (por tensor) e com
uma escala propria (por canal).

Uso:
  python fig_quant_scheme.py --ckpt results_macro/ckpt/macro_efficientnet_lite0_seed0_fp32.pt \
      --layer blocks.6.0.conv_dw.0 --out paper/figs/quant_scheme.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import weight_spread as W

C_PT, C_PC, INK, MUTED, GRID = "#eb6834", "#2a78d6", "#2b2b2a", "#8a8984", "#d9d8d3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", default="blocks.6.0.conv_dw.0")
    ap.add_argument("--n-classes", type=int, default=20)
    ap.add_argument("--out", default="paper/figs/quant_scheme.pdf")
    args = ap.parse_args()

    m = W.fused_model("efficientnet_lite0", args.n_classes, args.ckpt)
    w = dict(m.named_modules())[args.layer].weight.detach().flatten(1).numpy()
    amax = np.abs(w).max(1)
    alive = amax > 1e-6 * amax.max()
    order = np.argsort(amax)
    alive_idx = [c for c in order if alive[c]]
    picks = [("largest filter", order[-1]),
             ("median filter", alive_idx[len(alive_idx) // 2]),
             ("small filter (10th pct.)", alive_idx[len(alive_idx) // 10])]
    s_pt = amax.max() / 127

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(3, 2, figsize=(3.45, 3.0))
    for r, (lab, c) in enumerate(picks):
        x = w[c]
        lim = amax[c] * 1.15
        for col, (mode, color) in enumerate((("per-tensor", C_PT), ("per-channel", C_PC))):
            ax = axes[r, col]
            s = s_pt if mode == "per-tensor" else amax[c] / 127
            q = np.clip(np.round(x / s), -128, 127) * s
            n_lv = len(np.unique(np.round(x / s)))
            err = np.linalg.norm(q - x) / max(np.linalg.norm(x), 1e-12)
            # grade de quantizacao (so quando visivel)
            k = int(np.ceil(lim / s))
            if k <= 40:
                for g in np.arange(-k, k + 1) * s:
                    ax.axvline(g, color=GRID, lw=0.6, zorder=0)
            else:
                ax.axvspan(-lim, lim, color=GRID, alpha=0.25, lw=0, zorder=0)
            for xi, qi in zip(x, q):
                ax.plot([xi, qi], [1, 0], color=MUTED, lw=0.4, zorder=1)
            ax.scatter(x, np.ones_like(x), s=9, facecolor="white", edgecolor=INK, lw=0.6, zorder=2)
            ax.scatter(q, np.zeros_like(q), s=9, color=color, zorder=3)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-0.6, 1.6)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["INT8", "FP32"] if col == 0 else ["", ""])
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=5.5, length=2)
            ax.set_xlabel(f"{n_lv} INT8 level{'s' if n_lv > 1 else ''}, error {100 * err:.1f}%",
                          fontsize=6, color=color, labelpad=1)
            ax.set_title(lab if col == 0 else " ", fontsize=6.4, color=INK, loc="left", pad=2)
    fig.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0, 1, 0.93))
    for col, (mode, color) in enumerate((("Per-tensor scale (default)", C_PT), ("Per-channel scale", C_PC))):
        p = axes[0, col].get_position()
        fig.text((p.x0 + p.x1) / 2, 0.975, mode, ha="center", va="top", fontsize=7.5, color=color,
                 fontweight="bold")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    ratio = amax.max() / np.median(amax[alive])
    print(f"[ok] {out} | camada {args.layer}: {w.shape[0]} filtros, maior/mediana = {ratio:.1f}x, "
          f"escala por tensor = {s_pt:.4f}")


if __name__ == "__main__":
    main()
