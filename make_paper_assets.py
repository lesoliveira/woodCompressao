#!/usr/bin/env python3
"""
Tabelas e figuras do artigo (IJCNN), juntando as duas bases e as quatro arquiteturas.

Le results_macro/ e results_micro/ (results.csv, results_ablation.csv, layer_error_summary.csv)
e results_weight_spread/weight_spread_models.csv. Escreve em paper/tables/ e paper/figs/:

  tables/main.tex        FP32 e diferencas pareadas da PTQ por tensor e por canal (Tabela principal)
  tables/ablation.tex    ablacao da PTQ na MobileNetV3 (duas bases)
  tables/weights.tex     erro de quantizacao dos pesos por arquitetura
  tables/qat.tex         QAT por tensor e por canal contra o controle sem quantizacao
  figs/deltas.pdf        Delta FPR@95 por arquitetura e detector, por tensor x por canal
  numbers.json           numeros citados no texto

Uso:  python make_paper_assets.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
OUT = ROOT / "paper"
DS = [("macro", "Macro"), ("micro", "Micro")]
ARCH = [("mobilenet_v3_large", "MobileNetV3-L"), ("mobilenet_v2", "MobileNetV2"),
        ("efficientnet_lite0", "EfficientNet-Lite0"), ("resnet18", "ResNet-18")]
DETS = ["msp", "maxlogit", "energy", "knn", "mahalanobis"]
DET_LABEL = {"msp": "MSP", "maxlogit": "MaxLogit", "energy": "Energy", "knn": "kNN", "mahalanobis": "Mahal."}


def load():
    frames = []
    for ds, _ in DS:
        r = pd.read_csv(ROOT / f"results_{ds}" / "results.csv")
        a = pd.read_csv(ROOT / f"results_{ds}" / "results_ablation.csv")
        a = a.copy()
        a["variant"] = a.variant.replace({"abl_pc": "ptq_int8_pc"})
        # para as redes do torchvision, a PTQ por canal vem da ablacao (mesma calibracao)
        a = a[~((a.variant == "ptq_int8_pc") & a.backbone.isin(r[r.variant == "ptq_int8_pc"].backbone.unique()))]
        frames += [r, a]
    return pd.concat(frames, ignore_index=True)


def paired(df, ds, bb, a, b, det, metric):
    A = df[(df.dataset == ds) & (df.backbone == bb) & (df.variant == a) & (df.detector == det)].set_index("seed")[metric]
    B = df[(df.dataset == ds) & (df.backbone == bb) & (df.variant == b) & (df.detector == det)].set_index("seed")[metric]
    s = sorted(set(A.index) & set(B.index))
    # arredonda para que empates (|d| iguais) sejam tratados igual em qualquer ordem de operacoes
    return np.round(A.loc[s].to_numpy() * 100, 6), np.round(B.loc[s].to_numpy() * 100, 6)


def wil(d):
    return 1.0 if len(d) < 2 or np.allclose(d, 0) else stats.wilcoxon(d, zero_method="zsplit").pvalue


def star(p):
    return r"$^{\ddagger}$" if p < 0.01 else r"$^{\dagger}$" if p < 0.05 else ""


def dcell(df, ds, bb, a, b, det, metric, fmt="+.1f", worse_sign=1):
    A, B = paired(df, ds, bb, a, b, det, metric)
    if len(A) == 0:
        return "--", None
    d = np.round(A - B, 6)
    return f"{d.mean():{fmt}}{star(wil(d))}", dict(mean=float(d.mean()), std=float(d.std(ddof=1)),
                                                     worse=int((np.sign(d) == worse_sign).sum()),
                                                     n=len(d), p=float(wil(d)))


def mean_of(df, ds, bb, v, det, metric):
    x = df[(df.dataset == ds) & (df.backbone == bb) & (df.variant == v) & (df.detector == det)][metric]
    return float(x.mean() * 100) if len(x) else np.nan


def size_of(df, ds, bb, v):
    x = df[(df.dataset == ds) & (df.backbone == bb) & (df.variant == v) & (df.detector == "msp")].size_mb
    return float(x.mean()) if len(x) else np.nan


def table_main(df, nums):
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Closed-set accuracy and unknown-species detection (FPR@95, \%) of the FP32 models, and paired "
         r"differences (percentage points, mean over 10 seeds) caused by INT8 post-training quantization with "
         r"per-tensor weights (the PyTorch QNNPACK default) and with per-channel weights. Positive $\Delta$FPR@95 "
         r"means more unknown species accepted as known. Paired Wilcoxon test: $^{\dagger}p<0.05$, "
         r"$^{\ddagger}p<0.01$.}",
         r"\label{tab:main}", r"\setlength{\tabcolsep}{3.2pt}", r"\footnotesize",
         r"\begin{tabular}{ll ccc c cccc c cccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{FP32} & & \multicolumn{4}{c}{INT8, per-tensor weights (default)} & & "
         r"\multicolumn{4}{c}{INT8, per-channel weights} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){7-10}\cmidrule(lr){12-15}",
         r"Data & Backbone & Acc. & \multicolumn{2}{c}{FPR@95} & & $\Delta$Acc. & \multicolumn{2}{c}{$\Delta$FPR@95} & MB "
         r"& & $\Delta$Acc. & \multicolumn{2}{c}{$\Delta$FPR@95} & MB \\",
         r" & & & Energy & kNN & & & Energy & kNN & & & & Energy & kNN & \\", r"\midrule"]
    for i, (ds, dsl) in enumerate(DS):
        for j, (bb, bbl) in enumerate(ARCH):
            acc0 = mean_of(df, ds, bb, "fp32", "msp", "acc")
            e0, k0 = mean_of(df, ds, bb, "fp32", "energy", "fpr95"), mean_of(df, ds, bb, "fp32", "knn", "fpr95")
            cells = [dsl if j == 0 else "", bbl, f"{acc0:.1f}", f"{e0:.1f}", f"{k0:.1f}", ""]
            for v in ("ptq_int8", "ptq_int8_pc"):
                ca, na = dcell(df, ds, bb, v, "fp32", "msp", "acc", "+.2f", -1)
                ce, ne = dcell(df, ds, bb, v, "fp32", "energy", "fpr95")
                ck, nk = dcell(df, ds, bb, v, "fp32", "knn", "fpr95")
                cells += [ca, ce, ck, f"{size_of(df, ds, bb, v):.1f}", ""]
                for det in DETS:
                    _, nd = dcell(df, ds, bb, v, "fp32", det, "fpr95")
                    nums[f"{ds}/{bb}/{v}/fpr95/{det}"] = nd
                nums[f"{ds}/{bb}/{v}/acc"] = na
                nums[f"{ds}/{bb}/{v}/size_mb"] = size_of(df, ds, bb, v)
            nums[f"{ds}/{bb}/fp32/acc"] = acc0
            for det in DETS:
                nums[f"{ds}/{bb}/fp32/fpr95/{det}"] = mean_of(df, ds, bb, "fp32", det, "fpr95")
            nums[f"{ds}/{bb}/fp32/size_mb"] = size_of(df, ds, bb, "fp32")
            L.append(" & ".join(cells[:-1]) + r" \\")
        if i == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


ABL = [("base", "Default (per-tensor weights)"), ("pc", "Per-channel weights"),
       ("w_only_pt", "Weights only, per-tensor"), ("w_only_pc", "Weights only, per-channel"),
       ("fc32", "Last linear layer in FP32"), ("last3", "Last 3 blocks + classifier in FP32"),
       ("cal25", "Calibration with 25 images"), ("cal400", "Calibration with 400 images"),
       ("minmax", "MinMax activation observer")]


def table_ablation(df, nums, bb="mobilenet_v3_large"):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{PTQ ablation on MobileNetV3-L: paired differences to FP32 (pp, 10 seeds). "
         r"\emph{Weights only}: weights quantized to INT8, activations kept in FP32.}",
         r"\label{tab:ablation}", r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l ccc ccc}", r"\toprule",
         r" & \multicolumn{3}{c}{Macro} & \multicolumn{3}{c}{Micro} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"PTQ configuration & $\Delta$Acc. & \multicolumn{2}{c}{$\Delta$FPR@95} & $\Delta$Acc. & "
         r"\multicolumn{2}{c}{$\Delta$FPR@95} \\",
         r" & & Energy & kNN & & Energy & kNN \\", r"\midrule"]
    for cfg, lab in ABL:
        v = "ptq_int8_pc" if cfg == "pc" else f"abl_{cfg}"
        cells = [lab]
        for ds, _ in DS:
            ca, na = dcell(df, ds, bb, v, "fp32", "msp", "acc", "+.2f", -1)
            ce, ne = dcell(df, ds, bb, v, "fp32", "energy", "fpr95")
            ck, nk = dcell(df, ds, bb, v, "fp32", "knn", "fpr95")
            cells += [ca, ce, ck]
            nums[f"abl/{ds}/{bb}/{cfg}"] = dict(acc=na, energy=ne, knn=nk)
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)


def table_weights(nums):
    lay = pd.read_csv(ROOT / "results_weight_spread" / "weight_spread_layers.csv")
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{INT8 weight quantization error (BN folded), mean over layers, checkpoints and datasets. "
         r"Spread: largest over median per-channel weight range. Dead: channels (non-zero in FP32) whose weights "
         r"all round to zero with a per-tensor scale. Last column: $\Delta$FPR@95 (Energy) of default PTQ, "
         r"macro/micro.}",
         r"\label{tab:weights}", r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
         r"\begin{tabular}{l cccc c c}", r"\toprule",
         r" & \multicolumn{4}{c}{Depthwise layers} & All & \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}",
         r"Backbone & Spread & Err.\ PT & Err.\ PC & Dead & Err.\ PT & $\Delta$FPR@95 \\",
         r"\midrule"]
    for bb, bbl in ARCH:
        g = lay[lay.backbone == bb]
        dw = g[g.depthwise]
        per = lambda x, col: x.groupby(["dataset", "seed"])[col].mean().mean()
        rec = dict(err_pt_all=per(g, "err_pt"), err_pc_all=per(g, "err_pc"))
        if len(dw):
            rec.update(spread_dw=dw.groupby(["dataset", "seed"]).spread.median().mean(),
                       err_pt_dw=per(dw, "err_pt"), err_pc_dw=per(dw, "err_pc"), dead_dw=per(dw, "dead_frac"))
            cells = [f"{rec['spread_dw']:.1f}$\\times$", f"{100 * rec['err_pt_dw']:.1f}\\%",
                     f"{100 * rec['err_pc_dw']:.1f}\\%", f"{100 * rec['dead_dw']:.1f}\\%"]
        else:
            cells = ["--"] * 4
        dfp = "/".join(f"{nums[f'{ds}/{bb}/ptq_int8/fpr95/energy']['mean']:+.1f}" for ds, _ in DS)
        L.append(" & ".join([bbl, *cells, f"{100 * rec['err_pt_all']:.1f}\\%", dfp]) + r" \\")
        nums[f"weights/{bb}"] = {k: float(v) for k, v in rec.items()}
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_qat(df, nums):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{QAT against a control with the same fine-tuning and no quantization: paired $\Delta$FPR@95 "
         r"(pp, 10 seeds). Accuracy changed by at most 0.11\,pp in all cases.}",
         r"\label{tab:qat}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
         r"\begin{tabular}{ll cc cc}", r"\toprule",
         r" & & \multicolumn{2}{c}{QAT, per-tensor} & \multicolumn{2}{c}{QAT, per-channel} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Data & Backbone & Energy & kNN & Energy & kNN \\", r"\midrule"]
    for i, (ds, dsl) in enumerate(DS):
        for j, (bb, bbl) in enumerate(ARCH[:2]):
            cells = [dsl if j == 0 else "", bbl]
            for v in ("qat_int8_lr1e-4", "qat_int8_pc_lr1e-4"):
                for det in ("energy", "knn"):
                    c, n = dcell(df, ds, bb, v, "fp32_ft_lr1e-4", det, "fpr95")
                    cells.append(c)
                    nums[f"qat/{ds}/{bb}/{v}/{det}"] = n
                _, na = dcell(df, ds, bb, v, "fp32_ft_lr1e-4", "msp", "acc", "+.2f", -1)
                nums[f"qat/{ds}/{bb}/{v}/acc"] = na
            L.append(" & ".join(cells) + r" \\")
        if i == 0:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def fig_deltas(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.edgecolor": "#8a8984",
                         "axes.linewidth": 0.6})
    C_PT, C_PC, INK = "#eb6834", "#2a78d6", "#2b2b2a"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=False)
    for ax, (ds, dsl) in zip(axes, DS):
        xt = []
        for i, (bb, bbl) in enumerate(ARCH):
            for k, (v, col, off) in enumerate((("ptq_int8", C_PT, -0.18), ("ptq_int8_pc", C_PC, 0.18))):
                for d, det in enumerate(DETS):
                    A, B = paired(df, ds, bb, v, "fp32", det, "fpr95")
                    if len(A) == 0:
                        continue
                    dd = A - B
                    x = i + off + (d - 2) * 0.05
                    mk = "o" if det in ("msp", "maxlogit", "energy") else "s"
                    ax.errorbar(x, dd.mean(), yerr=dd.std(ddof=1) / np.sqrt(len(dd)), fmt=mk, ms=3.2,
                                color=col, mfc=col if mk == "o" else "white", mec=col, elinewidth=0.6,
                                capsize=0, lw=0)
            xt.append({"mobilenet_v3_large": "MobileNetV3-L", "mobilenet_v2": "MobileNetV2",
                       "efficientnet_lite0": "EffNet-Lite0", "resnet18": "ResNet-18"}[bb])
        ax.axhline(0, color="#8a8984", lw=0.6)
        ax.set_xticks(range(len(ARCH)))
        ax.set_xticklabels(xt)
        ax.set_title(f"{dsl} dataset", loc="left", color=INK)
        ax.set_ylabel(r"$\Delta$FPR@95 (pp)")
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C_PT, ms=4, label="INT8, per-tensor weights (default)"),
         Line2D([], [], marker="o", ls="", color=C_PC, ms=4, label="INT8, per-channel weights"),
         Line2D([], [], marker="o", ls="", color="#8a8984", ms=4, label="logit-based (MSP, MaxLogit, Energy)"),
         Line2D([], [], marker="s", ls="", mfc="white", mec="#8a8984", ms=4, label="feature-based (kNN, Mahal.)")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=250)
    plt.close(fig)


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    df = load()
    nums = {}
    (OUT / "tables" / "main.tex").write_text(table_main(df, nums))
    (OUT / "tables" / "ablation.tex").write_text(table_ablation(df, nums))
    (OUT / "tables" / "weights.tex").write_text(table_weights(nums))
    (OUT / "tables" / "qat.tex").write_text(table_qat(df, nums))
    fig_deltas(df, OUT / "figs" / "deltas.pdf")
    for ds, _ in DS:   # mecanismo (escores) e erro por bloco, ja gerados pelos outros scripts
        for src, dst in ((f"results_{ds}/paper/fig_mechanism.pdf", f"mechanism_{ds}.pdf"),
                         (f"results_{ds}/fig_layer_error.pdf", f"layer_error_{ds}.pdf")):
            if (ROOT / src).exists():
                (OUT / "figs" / dst).write_bytes((ROOT / src).read_bytes())
    for ds, _ in DS:
        m = pd.read_csv(ROOT / f"results_{ds}" / "paper" / "mechanism_summary.csv")
        nums[f"mech/{ds}"] = m.round(4).to_dict(orient="records")
        le = pd.read_csv(ROOT / f"results_{ds}" / "layer_error_summary.csv")
        nums[f"layer_error/{ds}"] = le[le.block == "logits"].round(4).to_dict(orient="records")
    (OUT / "numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    print(f"[ok] {OUT}")


if __name__ == "__main__":
    main()
