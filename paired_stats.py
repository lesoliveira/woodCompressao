#!/usr/bin/env python3
"""
Comparacoes pareadas por semente do Experimento 1.

Para cada backbone, detector e comparacao (A contra B), calcula a diferenca media A - B
em acuracia, AUROC e FPR@95 (em pontos percentuais), o desvio, o numero de sementes em
que A ficou pior/melhor que B, o teste t pareado e o teste de Wilcoxon pareado.
So entram sementes presentes nas duas variantes.

Comparacoes padrao:
  ptq_int8        contra fp32             efeito da PTQ
  qat_int8_lr1e-4 contra fp32_ft_lr1e-4   efeito da quantizacao na QAT, descontado o ajuste fino
  fp32_ft         contra fp32             efeito do ajuste fino extra com lr 1e-3 (sem quantizar)

Saida: paired_stats.csv na pasta do results.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PAIRS = [("ptq_int8", "fp32"), ("qat_int8_lr1e-4", "fp32_ft_lr1e-4"), ("fp32_ft", "fp32")]
METRICS = ["acc", "auroc", "fpr95"]
WORSE_IF_LOWER = {"acc": True, "auroc": True, "fpr95": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--pairs", nargs="*", default=None, help="pares A:B (padrao: os tres acima)")
    ap.add_argument("--out-name", default="paired_stats.csv", help="nome do CSV de saida")
    args = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else PAIRS

    df = pd.read_csv(args.results)
    rows = []
    for bb in sorted(df.backbone.unique()):
        for a, b in pairs:
            for det in sorted(df.detector.unique()):
                A = df[(df.backbone == bb) & (df.variant == a) & (df.detector == det)].set_index("seed")
                B = df[(df.backbone == bb) & (df.variant == b) & (df.detector == det)].set_index("seed")
                seeds = sorted(set(A.index) & set(B.index))
                if len(seeds) < 2:
                    continue
                r = {"dataset": A.dataset.iloc[0], "backbone": bb, "A": a, "B": b, "detector": det,
                     "n_seeds": len(seeds)}
                for m in METRICS:
                    d = (A.loc[seeds, m] - B.loc[seeds, m]).to_numpy() * 100
                    worse = (d < 0) if WORSE_IF_LOWER[m] else (d > 0)
                    better = (d > 0) if WORSE_IF_LOWER[m] else (d < 0)
                    r[f"{m}_diff_pp"] = round(d.mean(), 3)
                    r[f"{m}_std_pp"] = round(d.std(ddof=1), 3)
                    r[f"{m}_A_worse"] = int(worse.sum())
                    r[f"{m}_A_better"] = int(better.sum())
                    if np.allclose(d, 0):
                        r[f"{m}_p_t"] = r[f"{m}_p_wilcoxon"] = 1.0
                    else:
                        r[f"{m}_p_t"] = round(stats.ttest_rel(A.loc[seeds, m], B.loc[seeds, m]).pvalue, 4)
                        r[f"{m}_p_wilcoxon"] = round(stats.wilcoxon(d, zero_method="zsplit").pvalue, 4)
                rows.append(r)

    res = pd.DataFrame(rows)
    if res.empty:
        print("[aviso] nenhum par com pelo menos 2 sementes em comum: nada a comparar")
        return
    out = Path(args.results).parent / args.out_name
    res.to_csv(out, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = res[["backbone", "A", "B", "detector", "n_seeds",
                "acc_diff_pp", "acc_p_t",
                "auroc_diff_pp", "auroc_A_worse", "auroc_p_t",
                "fpr95_diff_pp", "fpr95_A_worse", "fpr95_p_t", "fpr95_p_wilcoxon"]]
    print("Diferencas A - B em pontos percentuais. *_A_worse = sementes em que A ficou pior que B.")
    print(show.to_string(index=False))
    print(f"\n[ok] {out}")


if __name__ == "__main__":
    main()
