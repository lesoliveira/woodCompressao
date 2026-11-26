#!/usr/bin/env python3
"""
Teste de robustez: as conclusoes mudam sem as imagens de teste que tem uma "gemea" no treino?

Algumas imagens da base sao repetidas ou mostram o mesmo campo/amostra com pequeno deslocamento
(grupos em groups/<base>_groups.csv, detectados por correlacao de fase). Numa divisao por imagem,
parte das conhecidas de teste tem uma imagem do mesmo grupo no treino. Este script recalcula as
metricas a partir dos escores salvos, excluindo essas imagens de teste, e compara as diferencas
pareadas (INT8 contra referencia) com e sem elas.

Uso:
  python check_leakage.py --results results_macro --groups groups/macro_groups.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DETS = ["msp", "maxlogit", "energy", "knn", "mahalanobis"]
PAIRS = [("ptq_int8", "fp32"), ("qat_int8_lr1e-4", "fp32_ft_lr1e-4")]


def fpr95(k, u):
    return float((u >= np.quantile(k, 0.05)).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--groups", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    with open(args.groups) as f:
        rows = list(csv.DictReader(f))
    g_by = {}
    for r in rows:                       # aceita chave pelo caminho relativo ou so pelo nome do arquivo
        g_by[r["file"]] = r["group"]
        g_by.setdefault(Path(r["file"]).name, r["group"])

    def grp(p):
        parts = Path(p).parts
        for i in range(len(parts)):
            k = str(Path(*parts[i:]))
            if k in g_by:
                return g_by[k]
        return "_" + p

    out = []
    for sp in sorted((res / "splits").glob("*_seed*.json")):
        split = json.load(open(sp))
        if split.get("grouped"):
            continue
        dname, seed = sp.stem.rsplit("_seed", 1)
        seed = int(seed)
        train_groups = {grp(p) for p, _ in split["train"]}
        leak = np.array([grp(p) in train_groups for p, _ in split["test_known"]])
        for f in sorted((res / "scores").glob(f"{dname}_s{seed}_*.npz")):
            rest = f.stem[len(f"{dname}_s{seed}_"):]
            bb = next((b for b in ("mobilenet_v3_large", "mobilenet_v2", "resnet18", "efficientnet_lite0") if rest.startswith(b)), None)
            if bb is None:
                continue
            v = rest[len(bb) + 1:]
            z = np.load(f)
            lab, lg = z["labels_known"], z["logits_known"]
            for subset, m in (("all", np.ones(len(leak), bool)), ("clean", ~leak)):
                acc = float((lg[m].argmax(1) == lab[m]).mean())
                for d in DETS:
                    k, u = z["known_" + d][m], z["unknown_" + d]
                    out.append(dict(dataset=dname, seed=seed, backbone=bb, variant=v, detector=d, subset=subset,
                                    n_test_known=int(m.sum()), n_leak=int(leak.sum()), acc=acc,
                                    fpr95=fpr95(k, u)))
    df = pd.DataFrame(out)
    print(f"imagens conhecidas de teste com gemea no treino: "
          f"{df[df.subset == 'all'].groupby('seed').n_leak.first().mean():.1f} por semente, de "
          f"{df[df.subset == 'all'].n_test_known.mean():.0f}")
    rows = []
    for (bb, d, sub), g in df.groupby(["backbone", "detector", "subset"]):
        for q, r in PAIRS:
            A, B = g[g.variant == q].set_index("seed"), g[g.variant == r].set_index("seed")
            s = sorted(set(A.index) & set(B.index))
            if not s:
                continue
            dd = (A.loc[s].fpr95 - B.loc[s].fpr95).to_numpy() * 100
            da = (A.loc[s].acc - B.loc[s].acc).to_numpy() * 100
            rows.append(dict(backbone=bb, comparison=f"{q} vs {r}", detector=d, subset=sub,
                             acc_ref=B.loc[s].acc.mean() * 100, d_acc_pp=da.mean(),
                             fpr95_ref=B.loc[s].fpr95.mean() * 100, d_fpr95_pp=dd.mean(),
                             seeds_worse=int((dd > 0).sum()),
                             p=1.0 if np.allclose(dd, 0) else stats.wilcoxon(dd, zero_method="zsplit").pvalue))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(res / "leakage_check.csv", index=False)
    w = t.pivot_table(index=["backbone", "comparison", "detector"], columns="subset",
                      values=["acc_ref", "fpr95_ref", "d_fpr95_pp", "p"])
    pd.set_option("display.width", 250)
    print(w.round(3).to_string())
    print(f"\n[ok] {res / 'leakage_check.csv'}")


if __name__ == "__main__":
    main()
