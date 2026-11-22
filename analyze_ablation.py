#!/usr/bin/env python3
"""
Analise do Experimento 2 (ablacao da PTQ).

Compara cada configuracao abl_<config> com o FP32 do Experimento 1, pareado por semente:
Delta acuracia, Delta AUROC e Delta FPR@95 por detector (pp), sementes em que piorou e
Wilcoxon. Resume tambem o erro relativo bloco a bloco (layer_error.csv), conhecidas contra
desconhecidas.

Saidas na pasta de resultados:
  ablation_summary.csv   uma linha por (backbone, config, detector)
  layer_error_summary.csv
  fig_layer_error.pdf/.png

Uso:
  python analyze_ablation.py --results results_macro
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ORDER = ["base", "pc", "minmax", "pc_minmax", "w_only_pt", "w_only_pc", "fc32", "pc_fc32", "cls32",
         "last1", "last3", "cal25", "cal400"]
LABEL = {"base": "PTQ (backend default)", "pc": "per-channel weights", "minmax": "MinMax activations",
         "pc_minmax": "per-channel + MinMax", "w_only_pt": "weights only (per-tensor)",
         "w_only_pc": "weights only (per-channel)", "fc32": "last linear in FP32",
         "pc_fc32": "per-channel + last linear FP32", "cls32": "classifier in FP32",
         "last1": "last block + classifier FP32", "last3": "last 3 blocks + classifier FP32",
         "cal25": "calibration: 25 images", "cal400": "calibration: 400 images"}


def wil(d):
    return 1.0 if np.allclose(d, 0) else stats.wilcoxon(d, zero_method="zsplit").pvalue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results_macro")
    args = ap.parse_args()
    res = Path(args.results)
    ref = pd.read_csv(res / "results.csv")
    ref = ref[ref.variant == "fp32"]
    abl = pd.read_csv(res / "results_ablation.csv")

    rows = []
    for (bb, v, det), g in abl.groupby(["backbone", "variant", "detector"]):
        r = ref[(ref.backbone == bb) & (ref.detector == det)].set_index("seed")
        g = g.set_index("seed")
        s = sorted(set(g.index) & set(r.index))
        if not s:
            continue
        row = dict(backbone=bb, config=v.replace("abl_", ""), detector=det, n_seeds=len(s),
                   size_mb=g.size_mb.mean(), latency_ms=g.latency_ms.mean())
        for m, worse in (("acc", -1), ("auroc", -1), ("fpr95", 1)):
            d = (g.loc[s, m] - r.loc[s, m]).to_numpy() * 100
            row[f"d_{m}_pp"] = d.mean()
            row[f"{m}_worse"] = int((np.sign(d) == worse).sum())
            row[f"{m}_p"] = wil(d)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["order"] = out.config.map({c: i for i, c in enumerate(ORDER)})
    out = out.sort_values(["backbone", "order", "detector"]).drop(columns="order")
    out.round(4).to_csv(res / "ablation_summary.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    wide = out.pivot_table(index=["backbone", "config"], columns="detector", values="d_fpr95_pp", sort=False)
    acc = out[out.detector == "msp"].set_index(["backbone", "config"])[["d_acc_pp", "acc_p", "size_mb"]]
    tab = acc.join(wide).reset_index()
    tab["order"] = tab.config.map({c: i for i, c in enumerate(ORDER)})
    tab = tab.sort_values(["backbone", "order"]).drop(columns="order")
    tab.insert(2, "description", tab.config.map(LABEL))
    print("Diferenca em relacao ao FP32 (pp): acuracia e FPR@95 por detector")
    print(tab.round(2).to_string(index=False))

    le_path = res / "layer_error.csv"
    if not le_path.exists():
        return
    le = pd.read_csv(le_path)
    ls = le.groupby(["backbone", "config", "block", "group"], sort=False).rel_err_mean.mean().unstack("group")
    ls["ratio_unknown_known"] = ls["unknown"] / ls["known"]
    ls = ls.reset_index()

    def blk_order(b):
        if b == "logits":
            return 10 ** 6
        return int(b.split(".")[-1]) if b.startswith("features.") else int(b.replace("layer", ""))
    ls["_o"] = ls.block.map(blk_order)
    ls = ls.sort_values(["backbone", "config", "_o"]).drop(columns="_o")
    ls.round(4).to_csv(res / "layer_error_summary.csv", index=False)
    print("\nErro relativo de quantizacao por bloco (media sobre sementes)")
    print(ls.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    bbs = list(ls.backbone.unique())
    fig, axes = plt.subplots(1, len(bbs), figsize=(3.5 * len(bbs), 2.6), squeeze=False)
    colors = {"known": "#2a78d6", "unknown": "#eb6834"}
    for ax, bb in zip(axes[0], bbs):
        for cfg, ls_ in (("base", "-"), ("pc", "--")):
            sub = ls[(ls.backbone == bb) & (ls.config == cfg)]
            if sub.empty:
                continue
            x = np.arange(len(sub))
            for grp in ("known", "unknown"):
                ax.plot(x, sub[grp].to_numpy(), ls_, color=colors[grp], lw=1.6,
                        label=f"{grp}, {'per-tensor' if cfg == 'base' else 'per-channel'}")
            ax.set_xticks(x)
            ax.set_xticklabels([b.replace("features.", "") for b in sub.block], fontsize=6, rotation=90)
        ax.set_title(bb, loc="left")
        ax.set_xlabel("Block")
        ax.set_ylabel("Relative quantization error")
        ax.grid(color="#e6e5e0", lw=0.5)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(res / f"fig_layer_error.{ext}", dpi=300)
    print(f"\n[ok] {res / 'ablation_summary.csv'}, {res / 'layer_error_summary.csv'}, fig_layer_error")


if __name__ == "__main__":
    main()
