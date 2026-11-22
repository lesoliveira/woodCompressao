#!/usr/bin/env python3
"""
Tabelas, figuras e analise de mecanismo do Experimento 1 (para o artigo).

Usa results.csv e os escores por imagem em scores/ (gerados com --save-scores).
Compara so os pares com controle correto:
  PTQ INT8          contra FP32
  QAT INT8 lr 1e-4  contra FP32 + ajuste fino lr 1e-4 (controle sem quantizacao)

Saidas em <results>/paper/:
  table_fpr95.tex, table_auroc.tex  resultados absolutos (media +- desvio), marcas de Wilcoxon
  table_paired.tex                  diferencas pareadas (pp) e sementes que pioraram
  fig_paired_deltas.pdf/.png        Delta acuracia e Delta FPR@95 por semente e detector
  fig_mechanism.pdf/.png            (a) escore FP32 x INT8 por imagem, (b) mudanca da distancia
                                    conhecidas-desconhecidas x Delta FPR@95
  mechanism.csv, mechanism_summary.csv
                                    por semente e detector: deslocamento medio dos escores de
                                    conhecidas e desconhecidas (em desvios), queda do limiar,
                                    desconhecidas novas aceitas/rejeitadas, d', correlacao de postos,
                                    ruido e FPR simulado com ruido gaussiano

Uso:
  python paper_figures.py --results results_macro
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DETS = ["msp", "maxlogit", "energy", "knn", "mahalanobis"]
DET_LABEL = {"msp": "MSP", "maxlogit": "MaxLogit", "energy": "Energy", "knn": "kNN", "mahalanobis": "Mahal."}
FAMILY = {"msp": "logit", "maxlogit": "logit", "energy": "logit", "knn": "feature", "mahalanobis": "feature"}
BB_LABEL = {"mobilenet_v3_large": "MobileNetV3-L", "resnet18": "ResNet-18", "mobilenet_v2": "MobileNetV2"}
VAR_LABEL = {"fp32": "FP32", "ptq_int8": "PTQ INT8", "fp32_ft_lr1e-4": "FP32 + FT", "qat_int8_lr1e-4": "QAT INT8"}
PAIRS = [("ptq_int8", "fp32", "PTQ vs. FP32"), ("qat_int8_lr1e-4", "fp32_ft_lr1e-4", "QAT vs. FT control")]
# paleta categorica validada (scripts/validate_palette.js do skill dataviz): azul, laranja
C_FEATURE, C_LOGIT, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#2b2b2a", "#8a8984"


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def wilcoxon_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return stats.wilcoxon(d, zero_method="zsplit").pvalue


def stars(p):
    return "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def paired(df, bb, a, b, det, metric):
    A = df[(df.backbone == bb) & (df.variant == a) & (df.detector == det)].set_index("seed")[metric]
    B = df[(df.backbone == bb) & (df.variant == b) & (df.detector == det)].set_index("seed")[metric]
    s = sorted(set(A.index) & set(B.index))
    return A.loc[s].to_numpy(), B.loc[s].to_numpy()


def table_absolute(df, metric, scale, fmt, caption, label, lower_better):
    bbs = [b for b in BB_LABEL if b in set(df.backbone)]
    base_of = {a: b for a, b, _ in PAIRS}
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\setlength{\tabcolsep}{3.5pt}", r"\footnotesize",
             r"\begin{tabular}{ll" + "c" * (2 + len(DETS)) + "}", r"\toprule",
             r"Backbone & Variant & MB & Acc. (\%) & " + " & ".join(DET_LABEL[d] for d in DETS) + r" \\",
             r"\midrule"]
    for bi, bb in enumerate(bbs):
        for vi, v in enumerate(VAR_LABEL):
            sub = df[(df.backbone == bb) & (df.variant == v)]
            if sub.empty:
                continue
            acc = sub[sub.detector == "msp"].acc * 100
            cells = [BB_LABEL[bb] if vi == 0 else "", VAR_LABEL[v], f"{sub.size_mb.mean():.1f}",
                     f"{acc.mean():.2f}$\\pm${acc.std():.2f}"]
            for d in DETS:
                x = sub[sub.detector == d][metric] * scale
                mark = ""
                if v in base_of:
                    a_, b_ = paired(df, bb, v, base_of[v], d, metric)
                    mark = stars(wilcoxon_p(a_, b_))
                cells.append(f"{x.mean():{fmt}}$\\pm${x.std():{fmt}}{mark}")
            lines.append(" & ".join(cells) + r" \\")
        if bi < len(bbs) - 1:
            lines.append(r"\midrule")
    n = df.seed.nunique()
    arrow = "lower" if lower_better else "higher"
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize Mean $\pm$ std over " + str(n) + r" seeds (random "
              r"known/unknown species splits); " + arrow + r" is better. FP32 + FT: same fine-tuning as QAT "
              r"without quantization. Paired Wilcoxon test against the uncompressed reference "
              r"(PTQ vs.\ FP32, QAT vs.\ FP32 + FT): $^{*}p<0.05$, $^{**}p<0.01$.}",
              r"\end{table}"]
    return "\n".join(lines)


def table_paired(df):
    bbs = [b for b in BB_LABEL if b in set(df.backbone)]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Paired differences caused by INT8 quantization (percentage points, mean over seeds; "
             r"in parentheses, seeds in which the quantized model was worse).}",
             r"\label{tab:paired}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
             r"\begin{tabular}{ll" + "c" * (1 + len(DETS)) + "}", r"\toprule",
             r" & & & \multicolumn{" + str(len(DETS)) + r"}{c}{$\Delta$FPR@95 (pp)} \\",
             r"\cmidrule(lr){4-" + str(3 + len(DETS)) + "}",
             r"Backbone & Comparison & $\Delta$Acc. (pp) & " + " & ".join(DET_LABEL[d] for d in DETS) + r" \\",
             r"\midrule"]
    for bi, bb in enumerate(bbs):
        for pi, (a, b, lab) in enumerate(PAIRS):
            A, B = paired(df, bb, a, b, "msp", "acc")
            if len(A) == 0:
                continue
            d = (A - B) * 100
            cells = [BB_LABEL[bb] if pi == 0 else "", lab,
                     f"{d.mean():+.2f}{stars(wilcoxon_p(A, B))} ({int((d < 0).sum())})"]
            for det in DETS:
                A, B = paired(df, bb, a, b, det, "fpr95")
                d = (A - B) * 100
                cells.append(f"{d.mean():+.1f}{stars(wilcoxon_p(A, B))} ({int((d > 0).sum())})")
            lines.append(" & ".join(cells) + r" \\")
        if bi < len(bbs) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize Paired Wilcoxon test: $^{*}p<0.05$, $^{**}p<0.01$.}",
              r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mecanismo (a partir dos escores por imagem)
# ---------------------------------------------------------------------------

def fpr95(k, u):
    return float((u >= np.quantile(k, 0.05)).mean())


def dprime(k, u):
    return (k.mean() - u.mean()) / np.sqrt((k.var() + u.var()) / 2)


def mechanism(scores_dir, df, rng):
    rows = []
    for bb in [b for b in BB_LABEL if b in set(df.backbone)]:
        for quant, ref, lab in PAIRS:
            for s in sorted(df.seed.unique()):
                fa = scores_dir / f"{df.dataset.iloc[0]}_s{s}_{bb}_{ref}.npz"    # A = referencia sem quantizar
                fb = scores_dir / f"{df.dataset.iloc[0]}_s{s}_{bb}_{quant}.npz"  # B = INT8
                if not (fa.exists() and fb.exists()):
                    continue
                A, B = np.load(fa), np.load(fb)
                for d in DETS:
                    ka, ua, kb, ub = A["known_" + d], A["unknown_" + d], B["known_" + d], B["unknown_" + d]
                    n = len(ka)
                    pa, pb = np.r_[ka, ua], np.r_[kb, ub]
                    if not (np.isfinite(pa).all() and np.isfinite(pb).all()) or pa.std() == 0 or pb.std() == 0:
                        continue                          # escores degenerados: nada a medir
                    za, zb = (pa - pa.mean()) / pa.std(), (pb - pb.mean()) / pb.std()
                    beta, alpha = np.polyfit(za, zb, 1)
                    res = zb - (alpha + beta * za)
                    sig = res.std()
                    sim = np.mean([fpr95(*np.split(alpha + beta * za + rng.normal(0, sig, len(za)), [n]))
                                   for _ in range(50)])
                    acc_a, acc_b = ua >= np.quantile(ka, 0.05), ub >= np.quantile(kb, 0.05)
                    low = za[:n] <= np.quantile(za[:n], 0.2)
                    sd_ref = np.sqrt((ka.var() + ua.var()) / 2)   # desvio da referencia (mesma escala de escore)
                    sk, su = (kb.mean() - ka.mean()) / sd_ref, (ub.mean() - ua.mean()) / sd_ref
                    rows.append(dict(
                        backbone=bb, comparison=lab, seed=s, detector=d, family=FAMILY[d],
                        rho_known=stats.spearmanr(ka, kb)[0], rho_unknown=stats.spearmanr(ua, ub)[0],
                        dprime_ref=dprime(ka, ua), dprime_int8=dprime(kb, ub),
                        noise_sigma=sig, noise_over_dprime=sig / dprime(za[:n], za[n:]),
                        fpr_ref=fpr95(ka, ua), fpr_int8=fpr95(kb, ub), fpr_sim_noise=sim,
                        newly_accepted=float((~acc_a & acc_b).mean()), newly_rejected=float((acc_a & ~acc_b).mean()),
                        tail_noise_ratio=res[:n][low].std() / res[n:].std(),
                        shift_known_sd=sk, shift_unknown_sd=su, gap_change_sd=sk - su,
                        thr_shift_sd=(np.quantile(kb, 0.05) - np.quantile(ka, 0.05)) / sd_ref))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": C_MUTED, "axes.linewidth": 0.6,
                         "xtick.color": C_INK, "ytick.color": C_INK, "axes.labelcolor": C_INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})


def fig_paired(df, out, plt):
    bbs = [b for b in BB_LABEL if b in set(df.backbone)]
    fig, axes = plt.subplots(len(bbs), len(PAIRS), figsize=(7.0, 2.2 * len(bbs)), sharey=True, squeeze=False)
    cats = ["acc"] + DETS
    rng = np.random.default_rng(1)
    for i, bb in enumerate(bbs):
        for j, (a, b, lab) in enumerate(PAIRS):
            ax = axes[i, j]
            ax.axhline(0, color=C_MUTED, lw=0.6, zorder=0)
            for x, c in enumerate(cats):
                if c == "acc":
                    A, B = paired(df, bb, a, b, "msp", "acc")
                    d, col = (A - B) * 100, C_MUTED
                else:
                    A, B = paired(df, bb, a, b, c, "fpr95")
                    d, col = (A - B) * 100, C_LOGIT if FAMILY[c] == "logit" else C_FEATURE
                jit = rng.uniform(-0.15, 0.15, len(d))
                ax.scatter(x + jit, d, s=12, color=col, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=2)
                ax.plot([x - 0.28, x + 0.28], [d.mean()] * 2, color=C_INK, lw=1.6, solid_capstyle="round", zorder=3)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(["$\\Delta$Acc."] + [DET_LABEL[d] for d in DETS])
            ax.axvline(0.5, color=C_MUTED, lw=0.4, ls=":")
            ax.set_title(f"{BB_LABEL[bb]}: {lab}", loc="left", color=C_INK)
            ax.grid(axis="y", color="#e6e5e0", lw=0.5)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Difference (pp)\n$\\Delta$Acc. or $\\Delta$FPR@95")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C_MUTED, label="Accuracy"),
         Line2D([], [], marker="o", ls="", color=C_LOGIT, label="Logit-based detector (FPR@95)"),
         Line2D([], [], marker="o", ls="", color=C_FEATURE, label="Feature-based detector (FPR@95)"),
         Line2D([], [], color=C_INK, lw=1.6, label="Mean over seeds")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_paired_deltas.{ext}", dpi=300)
    plt.close(fig)


def fig_mechanism(df, mech, scores_dir, out, plt, seed=0, bb="mobilenet_v3_large", det="energy"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    # (a) escore por imagem, FP32 x PTQ INT8
    A = np.load(scores_dir / f"{df.dataset.iloc[0]}_s{seed}_{bb}_fp32.npz")
    B = np.load(scores_dir / f"{df.dataset.iloc[0]}_s{seed}_{bb}_ptq_int8.npz")
    ka, ua, kb, ub = A["known_" + det], A["unknown_" + det], B["known_" + det], B["unknown_" + det]
    ax1.scatter(ua, ub, s=5, color=C_LOGIT, alpha=0.35, linewidth=0, label="Unknown species")
    ax1.scatter(ka, kb, s=6, color=C_FEATURE, alpha=0.6, linewidth=0, label="Known species")
    ta, tb = np.quantile(ka, 0.05), np.quantile(kb, 0.05)
    ax1.axvline(ta, color=C_INK, lw=0.7, ls="--")
    ax1.axhline(tb, color=C_INK, lw=0.7, ls="--")
    lo, hi = min(ua.min(), ub.min()), max(ka.max(), kb.max())
    ax1.plot([lo, hi], [lo, hi], color=C_MUTED, lw=0.6)
    newly = (ua < ta) & (ub >= tb)
    ax1.scatter(ua[newly], ub[newly], s=9, facecolor="none", edgecolor=C_INK, linewidth=0.6,
                label=f"Newly accepted unknowns ({newly.mean() * 100:.1f}%)")
    ax1.set_xlabel(f"{DET_LABEL[det]} score, FP32")
    ax1.set_ylabel(f"{DET_LABEL[det]} score, PTQ INT8")
    ax1.set_title(f"(a) {BB_LABEL[bb]}, seed {seed}: thresholds at 95% TPR (dashed)", loc="left")
    ax1.legend(loc="upper left", frameon=False, markerscale=1.5)
    # (b) mudanca da distancia entre conhecidas e desconhecidas x Delta FPR@95 (media sobre sementes).
    # MSP fica de fora: escore limitado em [0, 1] e saturado perto de 1, a escala de desvios nao se aplica.
    g = mech[mech.detector != "msp"].groupby(["backbone", "comparison", "detector", "family"]).agg(
        x=("gap_change_sd", "mean"), fr=("fpr_ref", "mean"), fi=("fpr_int8", "mean")).reset_index()
    g["dfpr"] = (g.fi - g.fr) * 100
    markers = {"mobilenet_v3_large": "o", "resnet18": "s", "mobilenet_v2": "D"}
    for _, r in g.iterrows():
        col = C_LOGIT if r.family == "logit" else C_FEATURE
        ax2.scatter(r.x, r.dfpr, s=34, marker=markers.get(r.backbone, "o"), linewidth=1.0, edgecolor=col,
                    facecolor="white" if r.comparison.startswith("QAT") else col, zorder=3)
    rho = stats.spearmanr(g.x, g.dfpr)[0]
    ax2.axhline(0, color=C_MUTED, lw=0.6)
    ax2.axvline(0, color=C_MUTED, lw=0.6)
    ax2.set_xlabel("Change in known-unknown mean gap (SD units)")
    ax2.set_ylabel("$\\Delta$FPR@95 (pp)")
    ax2.set_title(f"(b) All settings, MSP excluded (Spearman $\\rho$ = {rho:.2f})", loc="left")
    ax2.grid(color="#e6e5e0", lw=0.5)
    ax2.set_axisbelow(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C_LOGIT, label="Logit-based"),
         Line2D([], [], marker="o", ls="", color=C_FEATURE, label="Feature-based"),
         Line2D([], [], marker="o", ls="", mfc=C_MUTED, mec=C_MUTED, label="PTQ (filled)"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=C_MUTED, label="QAT (hollow)")]
    h += [Line2D([], [], marker=m, ls="", mfc="white", mec=C_INK, label=BB_LABEL[b])
          for b, m in markers.items() if b in set(g.backbone)]
    ax2.legend(handles=h, loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_mechanism.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results_macro", help="pasta com results.csv e scores/")
    ap.add_argument("--example-seed", type=int, default=0, help="semente do grafico (a) da figura de mecanismo")
    args = ap.parse_args()
    res = Path(args.results)
    out = res / "paper"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res / "results.csv")
    df = df[df.variant.isin(VAR_LABEL)]
    # so sementes com as quatro variantes, para as tabelas serem comparaveis
    full = df.groupby(["backbone", "seed"]).variant.nunique()
    keep = full[full == len(VAR_LABEL)].reset_index()[["backbone", "seed"]]
    df = df.merge(keep, on=["backbone", "seed"])
    print(f"[info] {df.seed.nunique()} sementes com as 4 variantes")

    (out / "table_fpr95.tex").write_text(table_absolute(
        df, "fpr95", 100, ".1f", "Closed-set accuracy and FPR@95 (\\%) for unknown-species detection.",
        "tab:fpr95", lower_better=True))
    (out / "table_auroc.tex").write_text(table_absolute(
        df, "auroc", 100, ".2f", "Closed-set accuracy and AUROC (\\%) for unknown-species detection.",
        "tab:auroc", lower_better=False))
    (out / "table_paired.tex").write_text(table_paired(df))

    mech = mechanism(res / "scores", df, np.random.default_rng(0))
    mech.to_csv(out / "mechanism.csv", index=False)
    summ = mech.groupby(["backbone", "comparison", "detector"]).mean(numeric_only=True).drop(columns="seed")
    summ["explained_by_noise"] = (summ.fpr_sim_noise - summ.fpr_ref) / (summ.fpr_int8 - summ.fpr_ref)
    pd.set_option("display.width", 250)
    cols = ["fpr_ref", "fpr_int8", "newly_accepted", "newly_rejected", "shift_known_sd", "shift_unknown_sd",
            "gap_change_sd", "thr_shift_sd", "dprime_ref", "dprime_int8", "rho_known", "rho_unknown",
            "noise_sigma", "fpr_sim_noise", "explained_by_noise"]
    print(summ[cols].round(3).to_string())
    summ[cols].round(4).to_csv(out / "mechanism_summary.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig_paired(df, out, plt)
    fig_mechanism(df, mech, res / "scores", out, plt, seed=args.example_seed)
    print(f"[ok] arquivos em {out}")


if __name__ == "__main__":
    main()
