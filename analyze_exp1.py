#!/usr/bin/env python3
"""
Analise do Experimento 1.

Gera:
  summary.csv        media e desvio por (base, backbone, variante, detector)
  drops.csv          perda relativa de acuracia e de AUROC em relacao ao FP32, por semente
  drops_summary.csv  media e desvio das perdas, com a diferenca (perda AUROC - perda acc)
  drops_scatter.png  perda de acuracia (x) contra perda de AUROC (y), com a diagonal
  qat_vs_control.csv QAT comparada ao seu controle sem quantizacao (fp32_ft com o mesmo sufixo),
                     separando o efeito da quantizacao do efeito do ajuste fino extra

Pontos acima da diagonal: a compressao prejudica mais a deteccao de
desconhecidas do que a classificacao fechada (hipotese H1).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["dataset", "backbone", "variant", "detector"]
METRICS = ["acc", "auroc", "fpr95", "oscr", "ece", "size_mb", "latency_ms"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results.csv gerado por run_exp1.py")
    ap.add_argument("--out", default=None, help="pasta de saida (padrao: a mesma do CSV)")
    args = ap.parse_args()
    out = Path(args.out or Path(args.results).parent)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results)
    summ = df.groupby(KEYS)[METRICS].agg(["mean", "std"]).round(4)
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ.reset_index().to_csv(out / "summary.csv", index=False)

    base = df[df.variant == "fp32"][["dataset", "seed", "backbone", "detector", "acc", "auroc", "oscr"]]
    base = base.rename(columns={"acc": "acc_fp32", "auroc": "auroc_fp32", "oscr": "oscr_fp32"})
    comp = df[df.variant != "fp32"].merge(base, on=["dataset", "seed", "backbone", "detector"])
    if comp.empty:
        print("[aviso] nao ha variantes comprimidas com FP32 correspondente: nada a comparar")
        return
    # perdas relativas em pontos percentuais do valor FP32 (positivo = piorou)
    comp["drop_acc_pct"] = 100 * (comp.acc_fp32 - comp.acc) / comp.acc_fp32
    # AUROC: perda relativa a distancia do acaso (0.5), para nao favorecer detectores fracos
    # detectores com AUROC FP32 < 0.55 ficam de fora (NaN): perda relativa nao faz sentido perto do acaso
    margin = (comp.auroc_fp32 - 0.5).where(comp.auroc_fp32 >= 0.55)
    comp["drop_auroc_pct"] = 100 * (comp.auroc_fp32 - comp.auroc) / margin
    comp["drop_oscr_pct"] = 100 * (comp.oscr_fp32 - comp.oscr) / comp.oscr_fp32.clip(lower=1e-6)
    comp["gap_pct"] = comp.drop_auroc_pct - comp.drop_acc_pct
    comp.to_csv(out / "drops.csv", index=False)

    ds = comp.groupby(KEYS)[["drop_acc_pct", "drop_auroc_pct", "drop_oscr_pct", "gap_pct"]] \
             .agg(["mean", "std"]).round(2)
    ds.columns = [f"{m}_{s}" for m, s in ds.columns]
    ds = ds.reset_index()
    ds.to_csv(out / "drops_summary.csv", index=False)

    # QAT x controle (fp32_ft): qat_int8 <-> fp32_ft, qat_int8_<tag> <-> fp32_ft_<tag>
    ctrl_rows = []
    for v in sorted(df.variant.unique()):
        if not v.startswith("qat_int8"):
            continue
        cv = "fp32_ft" + v[len("qat_int8"):]
        if cv not in set(df.variant):
            continue
        a = df[df.variant == v]
        b = df[df.variant == cv][["dataset", "seed", "backbone", "detector", "acc", "auroc", "fpr95"]]
        m = a.merge(b, on=["dataset", "seed", "backbone", "detector"], suffixes=("", "_ctrl"))
        m["variant_ctrl"] = cv
        ctrl_rows.append(m)
    if ctrl_rows:
        cc = pd.concat(ctrl_rows)
        cc["d_acc"] = cc.acc - cc.acc_ctrl
        cc["d_auroc"] = cc.auroc - cc.auroc_ctrl
        cc["d_fpr95"] = cc.fpr95 - cc.fpr95_ctrl
        cs = cc.groupby(KEYS + ["variant_ctrl"])[["acc_ctrl", "acc", "auroc_ctrl", "auroc", "fpr95_ctrl",
                                                  "fpr95", "d_acc", "d_auroc", "d_fpr95"]].mean().round(4)
        cs.reset_index().to_csv(out / "qat_vs_control.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n== Metricas (media sobre sementes) ==")
    print(df.groupby(KEYS)[["acc", "auroc", "fpr95", "oscr", "ece"]].mean().round(4).to_string())
    print("\n== Perdas relativas ao FP32 (%) ==")
    print(ds.to_string(index=False))
    if ctrl_rows:
        print("\n== QAT contra controle sem quantizacao (diferencas absolutas: QAT - controle) ==")
        print(cs.reset_index()[KEYS + ["d_acc", "d_auroc", "d_fpr95"]].to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib ausente: grafico nao gerado")
        return
    dets = sorted(comp.detector.unique())
    markers = {v: ("o" if v.startswith("ptq") else "^" if v.startswith("qat") else "D" if v.startswith("fp32_ft")
                   else "s") for v in sorted(comp.variant.unique())}
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    g = comp.groupby(KEYS)[["drop_acc_pct", "drop_auroc_pct"]].mean().reset_index()
    for (_, r) in g.iterrows():
        ax.scatter(r.drop_acc_pct, r.drop_auroc_pct, color=cmap(dets.index(r.detector) % 10),
                   marker=markers[r.variant], s=60, edgecolor="black", linewidth=0.5)
    lim = max(1.0, np.nanmax(np.abs(g[["drop_acc_pct", "drop_auroc_pct"]].values)) * 1.1)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Perda relativa de acuracia fechada (%)")
    ax.set_ylabel("Perda relativa de AUROC (% da distancia ao acaso)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=cmap(i % 10), marker="o", ls="", label=d) for i, d in enumerate(dets)]
    h += [Line2D([], [], color="black", marker=m, ls="", mfc="none", label=v) for v, m in markers.items()]
    ax.legend(handles=h, fontsize=8, loc="upper left")
    ax.set_title("Compressao: acuracia x deteccao de desconhecidas")
    fig.tight_layout()
    fig.savefig(out / "drops_scatter.png", dpi=200)
    print(f"\n[ok] arquivos em {out}")


if __name__ == "__main__":
    main()
