"""Experimento 6: as camadas depthwise sao a causa? Compara, contra o FP32, as configuracoes que
quantizam (ou trocam o esquema de) so as depthwise ou so as demais camadas.

Le results_{macro,micro}/results.csv (FP32) e results_ablation.csv (configuracoes abl_*).
Saida: results_dw/dw_isolation.csv e um resumo no terminal.

    python analyze_dw.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CONFIGS = [  # (variante, descricao)
    ("abl_base", "INT8, por tensor em tudo (padrao)"),
    ("abl_dw_pt", "INT8, por tensor so nas depthwise"),
    ("abl_dw_pc", "INT8, por canal so nas depthwise"),
    ("abl_pc", "INT8, por canal em tudo"),
    ("abl_w_only_pt", "so pesos, por tensor em tudo"),
    ("abl_w_only_dw", "so pesos, por tensor so nas depthwise"),
    ("abl_w_only_nodw", "so pesos, por tensor menos nas depthwise"),
    ("abl_w_only_stem", "so pesos, por tensor so na conv de entrada"),
    ("abl_w_only_pw", "so pesos, por tensor so nas pointwise 1x1"),
    ("abl_w_only_se", "so pesos, por tensor so no squeeze-excitation"),
    ("abl_w_only_head", "so pesos, por tensor so na ultima conv e no classificador"),
    ("abl_w_only_none", "nenhum peso quantizado (FP32 no dispositivo dos so-pesos)"),
]
# configuracoes de pesos simulados novas: comparadas com a abl_w_only_none (mesmo dispositivo de
# avaliacao). As demais (INT8 e w_only_pt da ablacao original) com o FP32 do Experimento 1 (CPU).
SAME_DEVICE_REF = {"abl_w_only_dw", "abl_w_only_nodw", "abl_w_only_stem", "abl_w_only_pw",
                   "abl_w_only_se", "abl_w_only_head"}
NETS = ["mobilenet_v3_large", "mobilenet_v2"]
DETS = ["msp", "maxlogit", "energy", "knn", "mahalanobis"]


def main():
    rows = []
    for ds in ("macro", "micro"):
        r = pd.read_csv(f"results_{ds}/results.csv")
        a = pd.read_csv(f"results_{ds}/results_ablation.csv")
        ref = r[r.variant == "fp32"].set_index(["backbone", "detector", "seed"]).sort_index()
        for bb in NETS:
            for d in DETS:
                base = None
                for v, desc in CONFIGS:
                    q = a[(a.backbone == bb) & (a.detector == d) & (a.variant == v)].set_index("seed")
                    if q.empty:
                        continue
                    f = ref.loc[(bb, d)]
                    refname = "fp32"
                    if v in SAME_DEVICE_REF:
                        nn_ = a[(a.backbone == bb) & (a.detector == d) & (a.variant == "abl_w_only_none")]
                        if not nn_.empty:
                            f = nn_.set_index("seed")
                            refname = "w_only_none"
                    s = sorted(set(q.index) & set(f.index))
                    dfpr = ((q.loc[s, "fpr95"] - f.loc[s, "fpr95"]) * 100).round(6).to_numpy()
                    dacc = ((q.loc[s, "acc"] - f.loc[s, "acc"]) * 100).to_numpy()
                    p = wilcoxon(dfpr).pvalue if np.any(dfpr != 0) else 1.0
                    m = float(dfpr.mean())
                    if v == "abl_base":
                        base = m
                    rows.append(dict(dataset=ds, backbone=bb, detector=d, variant=v, config=desc, n=len(s),
                                     reference=refname,
                                     d_fpr95=round(m, 2), p=float(p), d_acc=round(float(dacc.mean()), 2),
                                     share_of_base=round(m / base, 2) if base and abs(base) > 0.5 else np.nan))
    out = Path("results_dw")
    out.mkdir(exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "dw_isolation.csv", index=False)
    for d in ("energy", "knn"):
        print(f"\n=== Delta FPR@95 (pp) contra FP32, {d}, media de 10 sementes (* p<0,05) ===")
        x = t[t.detector == d].copy()
        x["val"] = x.d_fpr95.map("{:+.1f}".format) + np.where(x.p < 0.05, "*", " ")
        print(x.pivot_table(index="config", columns=["dataset", "backbone"], values="val", aggfunc="first",
                            sort=False).reindex([c for _, c in CONFIGS]).to_string())
    print(f"\n[ok] {out / 'dw_isolation.csv'}")


if __name__ == "__main__":
    main()
